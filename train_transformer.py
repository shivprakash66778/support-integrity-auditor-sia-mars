#!/usr/bin/env python3
"""
SIA — DistilBERT Fine-Tuning (Colab GPU)
==========================================
MARS Open Projects 2026 · Problem Statement 1

Fine-tunes distilbert-base-uncased on pseudo-labeled mismatch data.
Includes: metadata fusion, WeightedRandomSampler, class-weighted loss,
threshold tuning on validation set, leakage checks.

Usage (run on Colab with GPU):
  python train_transformer.py \
      --input_csv data/customer_support_tickets.csv \
      --output_dir outputs --model_dir model_artifacts \
      --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse, json, os, sys, warnings, time, tempfile, shutil
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _DIR not in sys.path: sys.path.insert(0, _DIR)

from train_pipeline import (
    ensure_dirs, generate_pseudo_labels, fit_resolution_thresholds,
    evaluate, RANDOM_STATE, PRIORITY_MAP,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
except ImportError:
    print("Install: pip install torch transformers")
    sys.exit(1)


SEED = RANDOM_STATE


# ═══════════════════════════════════════════════════════════════
# INPUT FORMATTING — Text + Structured Metadata
# ═══════════════════════════════════════════════════════════════

def build_transformer_input(row):
    """Fuse ticket text with structured metadata.
    PDF requirement: 'Input features must include text fields and at
    least one structured metadata feature.'
    """
    subject = str(row.get("Ticket_Subject", ""))
    desc = str(row.get("Ticket_Description", ""))
    category = str(row.get("Issue_Category", ""))
    priority = str(row.get("Priority_Level", ""))
    channel = str(row.get("Ticket_Channel", ""))
    resolution = str(row.get("Resolution_Time_Hours", ""))
    satisfaction = str(row.get("Satisfaction_Score", ""))

    return (
        f"[PRIORITY={priority}] "
        f"[CATEGORY={category}] "
        f"[CHANNEL={channel}] "
        f"[RESOLUTION_HOURS={resolution}] "
        f"[SATISFACTION={satisfaction}] "
        f"Subject: {subject} Description: {desc}"
    )


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class SIATransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts   = list(texts)
        self.labels  = list(labels)
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]), max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════
# SAMPLING & LOSS
# ═══════════════════════════════════════════════════════════════

def make_weighted_sampler(labels):
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=2)
    weights_per_class = 1.0 / np.maximum(class_counts, 1)
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(labels), replacement=True,
    )


def make_class_weights(labels):
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=2)
    total = class_counts.sum()
    weights = total / (2.0 * np.maximum(class_counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def predict_proba_loader(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y    = batch["labels"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            p = torch.softmax(out.logits.float(), dim=1)[:, 1]
            probs.extend(p.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())
    return np.array(probs), np.array(labels)


def metrics_from_probs(y_true, probs, threshold=0.5, model_name="Transformer"):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "precision_consistent": float(precision_score(y_true, preds, pos_label=0, zero_division=0)),
        "precision_mismatch": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall_consistent": float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
        "recall_mismatch": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(input_csv, output_dir, model_dir,
         model_name="distilbert-base-uncased",
         epochs=3, batch_size=16, lr=2e-5, max_len=128):

    ensure_dirs(output_dir, model_dir)
    distilbert_dir = os.path.join(model_dir, "distilbert_finetuned")
    ensure_dirs(distilbert_dir)

    print("=" * 65)
    print("  SIA — DistilBERT Fine-Tuning")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load & pseudo-label
    print("\n[1/7] Loading data & generating pseudo-labels...")
    df = pd.read_csv(input_csv)
    res_thresh = fit_resolution_thresholds(df["Resolution_Time_Hours"])
    df = generate_pseudo_labels(df, res_thresh)
    df_conf = df[df["mismatch_label"] != -1].copy().reset_index(drop=True)
    print(f"  Total: {len(df)}, Confident: {len(df_conf)}")
    print(f"  Class 0: {(df_conf['mismatch_label']==0).sum()}, Class 1: {(df_conf['mismatch_label']==1).sum()}")

    # 2. Build transformer inputs
    print("\n[2/7] Building transformer inputs...")
    df_conf["transformer_input"] = df_conf.apply(build_transformer_input, axis=1)

    # 3. Split 70/15/15
    print("\n[3/7] Splitting data 70/15/15...")
    train_df, temp_df = train_test_split(
        df_conf, test_size=0.30, random_state=SEED,
        stratify=df_conf["mismatch_label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df["mismatch_label"])
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Leakage check: no overlap
    train_ids = set(train_df.index)
    val_ids = set(val_df.index)
    test_ids = set(test_df.index)
    assert len(train_ids & val_ids) == 0, "Train-Val overlap!"
    assert len(train_ids & test_ids) == 0, "Train-Test overlap!"
    assert len(val_ids & test_ids) == 0, "Val-Test overlap!"
    print("  Split overlap check: PASSED (0 overlap)")

    # 4. Tokenizer, datasets, loaders
    print(f"\n[4/7] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params/1e6:.1f}M")

    train_labels = train_df["mismatch_label"].values
    class_weights = make_class_weights(train_labels).to(DEVICE)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    train_ds = SIATransformerDataset(train_df["transformer_input"], train_df["mismatch_label"], tokenizer, max_len)
    val_ds   = SIATransformerDataset(val_df["transformer_input"],   val_df["mismatch_label"],   tokenizer, max_len)
    test_ds  = SIATransformerDataset(test_df["transformer_input"],  test_df["mismatch_label"],  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=make_weighted_sampler(train_labels))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=False)

    # Loss, optimizer, scheduler
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.10 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5. Train
    print(f"\n[5/7] Training {model_name} for {epochs} epochs...")
    best_val_f1 = -1
    best_dir = tempfile.mkdtemp()
    no_improve = 0
    patience = 2
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl  = batch["labels"].to(DEVICE)

            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits.float(), lbl)
            loss.backward()
            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_probs, val_labels = predict_proba_loader(model, val_loader)
        val_m = metrics_from_probs(val_labels, val_probs, threshold=0.5)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, len(train_loader)),
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_recall_consistent": val_m["recall_consistent"],
            "val_recall_mismatch": val_m["recall_mismatch"],
            "seconds": elapsed,
        }
        history.append(row)
        print(f"  Epoch {epoch}/{epochs} | loss={row['loss']:.4f} | "
              f"acc={row['val_accuracy']:.4f} F1={row['val_macro_f1']:.4f} "
              f"Rc={row['val_recall_consistent']:.4f} Rm={row['val_recall_mismatch']:.4f} | {elapsed:.0f}s")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            no_improve = 0
            print("    Saved best checkpoint.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping after {patience} epochs without improvement.")
                break

    # Load best checkpoint
    print(f"\n  Loading best checkpoint (val F1={best_val_f1:.4f})...")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, num_labels=2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(best_dir)

    # 6. Threshold tuning on validation
    print("\n[6/7] Threshold tuning on validation set...")
    val_probs, val_labels = predict_proba_loader(model, val_loader)

    best_threshold = 0.5
    best_score = -1
    for t in np.round(np.arange(0.05, 0.96, 0.005), 3):
        m = metrics_from_probs(val_labels, val_probs, threshold=t)
        if m["macro_f1"] > best_score:
            best_score = m["macro_f1"]
            best_threshold = float(t)
    print(f"  Best threshold: {best_threshold}")

    # 7. Final test evaluation
    print("\n[7/7] Final test evaluation...")
    test_probs, test_labels = predict_proba_loader(model, test_loader)

    default_metrics = metrics_from_probs(test_labels, test_probs, threshold=0.5, model_name="DistilBERT (default)")
    tuned_metrics = metrics_from_probs(test_labels, test_probs, threshold=best_threshold, model_name="DistilBERT (tuned)")

    print(f"\n  Default threshold (0.5):")
    print(f"    Acc={default_metrics['accuracy']:.4f}  F1={default_metrics['macro_f1']:.4f}  "
          f"Rc={default_metrics['recall_consistent']:.4f}  Rm={default_metrics['recall_mismatch']:.4f}")
    print(f"  Tuned threshold ({best_threshold}):")
    print(f"    Acc={tuned_metrics['accuracy']:.4f}  F1={tuned_metrics['macro_f1']:.4f}  "
          f"Rc={tuned_metrics['recall_consistent']:.4f}  Rm={tuned_metrics['recall_mismatch']:.4f}")
    print(f"    CM: {tuned_metrics['confusion_matrix']}")

    passed = (
        tuned_metrics["accuracy"] >= 0.83 and
        tuned_metrics["macro_f1"] >= 0.82 and
        tuned_metrics["recall_consistent"] >= 0.78 and
        tuned_metrics["recall_mismatch"] >= 0.78
    )
    print(f"  MARS verification: {'PASSED' if passed else 'FAILED'}")

    # Leakage check on model inputs
    train_texts = list(train_df["transformer_input"])
    test_texts = list(test_df["transformer_input"])
    for forbidden in ["mismatch_label","mismatch_type","severity_delta","inferred_severity","final_decision","target"]:
        found = any(forbidden.lower() in str(x).lower() for x in train_texts[:100])
        print(f"  Leakage check '{forbidden}': {'FOUND' if found else 'clean'}")
    overlap = len(set(train_texts) & set(test_texts))
    print(f"  Train-test text overlap: {overlap}")

    # Save model
    if os.path.exists(distilbert_dir):
        shutil.rmtree(distilbert_dir)
    model.save_pretrained(distilbert_dir)
    tokenizer.save_pretrained(distilbert_dir)

    # Save metrics
    metrics_payload = {
        "model_name": model_name,
        "selected_threshold": best_threshold,
        "default_threshold_metrics": default_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "training_history": history,
        "mars_threshold_passed": passed,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "note": "Evaluation is on held-out pseudo-labeled test data, not human-verified mismatch labels.",
    }
    with open(os.path.join(output_dir, "transformer_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Update model_config.json if transformer beats existing
    cfg_path = os.path.join(model_dir, "model_config.json")
    if passed:
        with open(cfg_path, "w") as f:
            json.dump({
                "selected_model": "distilbert",
                "distilbert_threshold": best_threshold,
                "distilbert_model_dir": "distilbert_finetuned",
            }, f, indent=2)
        print(f"\n  model_config.json updated: selected_model = distilbert")

    # Confusion matrix plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    cm = np.array(tuned_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Consistent","Mismatch"], yticklabels=["Consistent","Mismatch"],
        ax=ax, annot_kws={"size":14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"DistilBERT Confusion Matrix (threshold={best_threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_distilbert.png"), dpi=150)
    plt.close()

    # Clean up
    shutil.rmtree(best_dir, ignore_errors=True)

    print(f"\n{'='*65}")
    print(f"  DONE — {model_name}")
    print(f"  Threshold: {best_threshold}")
    print(f"  Acc={tuned_metrics['accuracy']:.4f}  F1={tuned_metrics['macro_f1']:.4f}")
    print(f"  MARS: {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*65}")
    return tuned_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",   required=True)
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--model_dir",   default="model_artifacts")
    p.add_argument("--model_name",  default="distilbert-base-uncased")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max_len",     type=int,   default=128)
    a = p.parse_args()
    main(a.input_csv, a.output_dir, a.model_dir,
         a.model_name, a.epochs, a.batch_size, a.lr, a.max_len)
