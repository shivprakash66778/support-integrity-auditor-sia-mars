#!/usr/bin/env python3
"""
SIA — Prediction Script
Supports: DistilBERT (primary), MLP, LR, HGB (auto-detected from model_config.json)
Gracefully falls back if transformer artifacts are missing.
"""
import argparse, json, os, sys, warnings
import joblib, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _DIR not in sys.path: sys.path.insert(0, _DIR)

from train_pipeline import (
    PRIORITY_MAP, SEVERITY_LABELS, CATEGORY_SEVERITY,
    W_RULE, W_RESOLUTION, W_CATEGORY, W_SATISFACTION,
    clean_text, combine_text_fields, compute_text_stats,
    compute_rule_based_severity, compute_resolution_severity,
    compute_category_severity, compute_satisfaction_severity, fuse_severity,
    hybrid_decision, generate_dossier, get_matched_keywords,
    ensure_dirs, build_features, _build_dense_features,
    SIGNAL_COLS, TEXT_STAT_COLS, DENSE_FEATURE_COLS,
)

# Transformer input builder (matches train_transformer.py)
def build_transformer_input(row):
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


def load_artifacts(model_dir):
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_artifacts not found: {model_dir}")

    lr     = joblib.load(os.path.join(model_dir, "baseline_logistic_model.pkl"))
    tfidf  = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    meta   = joblib.load(os.path.join(model_dir, "metadata_preprocessor.pkl"))
    thresh = joblib.load(os.path.join(model_dir, "resolution_thresholds.pkl"))

    mlp = hgb = ohe = scaler = bert_model = bert_tokenizer = None
    bert_threshold = 0.5
    selected = "lr"

    # Load MLP if available
    mlp_candidates = [
        os.path.join(model_dir, "tfidf_mlp_model.pkl"),
        os.path.join(model_dir, "mlp_classifier.pkl"),
    ]
    
    for mlp_path in mlp_candidates:
        if os.path.exists(mlp_path):
            try:
                mlp = joblib.load(mlp_path)
                print(f"  MLP loaded from {mlp_path}")
                break
            except Exception as e:
                print(f"  MLP load failed from {mlp_path}: {e}")
                mlp = None

    cfg_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        selected = cfg.get("selected_model", "lr")
        bert_threshold = cfg.get("distilbert_threshold", 0.5)

        if selected == "distilbert":
            bert_dir = cfg.get("distilbert_model_dir", "distilbert_finetuned")
            bert_path = os.path.join(model_dir, bert_dir)
            if os.path.isdir(bert_path):
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
                    print(f"  DistilBERT loaded from {bert_path} (threshold={bert_threshold})")
                except Exception as e:
                    print(f"  DistilBERT load failed ({e}), falling back to MLP/LR")
                    selected = "mlp" if mlp else "lr"
            else:
                print(f"  DistilBERT dir not found: {bert_path}, falling back")
                selected = "mlp" if mlp else "lr"

        if selected == "hgb":
            try:
                hgb    = joblib.load(os.path.join(model_dir, "hgb_model.pkl"))
                ohe    = joblib.load(os.path.join(model_dir, "hgb_ohe.pkl"))
                scaler = joblib.load(os.path.join(model_dir, "hgb_scaler.pkl"))
            except Exception:
                selected = "mlp" if mlp else "lr"

    return lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bert_model, bert_tokenizer, selected, bert_threshold


def _bert_probs(texts, model, tokenizer, batch_size=32, threshold=0.5):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).logits
            probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def predict_batch(df, lr, tfidf, meta_prep, res_thresholds,
                  mlp=None, hgb=None, ohe=None, scaler=None,
                  bert_model=None, bert_tokenizer=None,
                  bert_threshold=0.5):
    df = df.copy()
    df["combined_text"] = df.apply(combine_text_fields, axis=1)
    df["clean_text"]    = df["combined_text"].apply(clean_text)
    df["rule_severity"]         = df["combined_text"].apply(compute_rule_based_severity)
    df["resolution_severity"]   = df["Resolution_Time_Hours"].apply(lambda h: compute_resolution_severity(h, res_thresholds))
    df["category_severity"]     = df["Issue_Category"].apply(compute_category_severity)
    df["satisfaction_severity"]  = df["Satisfaction_Score"].apply(compute_satisfaction_severity)

    fused = df.apply(lambda r: fuse_severity(
        r["rule_severity"], r["resolution_severity"],
        r["category_severity"], r["satisfaction_severity"]), axis=1)
    df["inferred_severity_score"] = fused.apply(lambda x: x[0])
    df["inferred_severity_level"] = fused.apply(lambda x: x[1])
    df["inferred_severity"]       = df["inferred_severity_level"].map(SEVERITY_LABELS)
    df["assigned_priority_score"] = df["Priority_Level"].map(PRIORITY_MAP)
    df["severity_delta"]          = df["inferred_severity_level"] - df["assigned_priority_score"]
    df["mismatch_type"] = df["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d>=2 else ("False Alarm" if d<=-2 else ("Consistent" if d==0 else "Borderline")))

    stats = df["combined_text"].apply(compute_text_stats).apply(pd.Series)
    for c in stats.columns: df[c] = stats[c]

    # Model inference — priority: DistilBERT > MLP > HGB > LR
    if bert_model is not None and bert_tokenizer is not None:
        transformer_texts = df.apply(build_transformer_input, axis=1).values
        df["mismatch_probability"] = _bert_probs(
            transformer_texts, bert_model, bert_tokenizer,
            threshold=bert_threshold)
    elif mlp is not None:
        X, _, _ = build_features(df, tfidf=tfidf, meta_prep=meta_prep)
        df["mismatch_probability"] = mlp.predict_proba(X)[:, 1]
    elif hgb is not None and ohe is not None and scaler is not None:
        X, _, _ = build_features(df, tfidf=tfidf, meta_prep=meta_prep)
        lr_probs = lr.predict_proba(X)[:, 1]
        X_d, _, _ = _build_dense_features(df, ohe=ohe, scaler=scaler)
        df["mismatch_probability"] = 0.4*lr_probs + 0.6*hgb.predict_proba(X_d)[:,1]
    else:
        X, _, _ = build_features(df, tfidf=tfidf, meta_prep=meta_prep)
        df["mismatch_probability"] = lr.predict_proba(X)[:, 1]

    decisions = df.apply(hybrid_decision, axis=1)
    df["final_decision"]   = decisions.apply(lambda x: x[0])
    df["decision_source"]  = decisions.apply(lambda x: x[1])
    df["final_confidence"] = decisions.apply(lambda x: x[2])
    return df


def main(input_csv, output_dir, model_dir):
    ensure_dirs(output_dir)
    lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bm, bt, sel, bt_thresh = load_artifacts(model_dir)
    print(f"SIA Prediction — Model: {sel.upper()}")
    df = pd.read_csv(input_csv)
    result = predict_batch(df, lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bm, bt, bt_thresh)
    mm = (result["final_decision"]=="Mismatch").sum()
    print(f"  {len(result)} tickets -> {mm} Mismatch, {len(result)-mm} Consistent")

    cols = [c for c in ["Ticket_ID","Ticket_Subject","Priority_Level","Issue_Category",
        "inferred_severity","severity_delta","mismatch_type","mismatch_probability",
        "final_decision","decision_source","final_confidence"] if c in result.columns]
    result[cols].to_csv(os.path.join(output_dir, "sia_predictions.csv"), index=False)
    mm_df = result[result["final_decision"]=="Mismatch"]
    dossiers = [generate_dossier(row) for _, row in mm_df.iterrows()]
    with open(os.path.join(output_dir, "sia_dossiers.json"), "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"  Saved: sia_predictions.csv, sia_dossiers.json ({len(dossiers)} dossiers)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", default="prediction_outputs")
    p.add_argument("--model_dir", default="model_artifacts")
    a = p.parse_args()
    main(a.input_csv, a.output_dir, a.model_dir)
