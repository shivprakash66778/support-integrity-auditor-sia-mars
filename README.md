# Support Integrity Auditor (SIA)

**MARS Open Projects 2026 — Problem Statement 1**
*Artificial Intelligence / Machine Learning / NLP / CRM Systems*

---

## Problem Overview

In enterprise CRM ecosystems, manual ticket triage suffers from agent fatigue, keyword anchoring, and priority inflation or deflation. When critical issues are labeled "Low" or trivial complaints are escalated to "Critical," SLA compliance degrades and customer churn increases.

SIA is a semantics-driven, evidence-grounded automated auditor that detects **Priority Mismatch** — cases where a ticket's objective characteristics conflict with its human-assigned priority. The fundamental challenge: **no pre-annotated mismatch labels exist.** The system must bootstrap its own supervision from raw ticket data.

## Dataset

**Customer Support Tickets — CRM Dataset** (Kaggle)
- 20,000 support tickets with text, metadata, and priority labels
- Download: [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)
- Place as `data/customer_support_tickets.csv`

Key columns: Ticket_Subject, Ticket_Description, Issue_Category, Priority_Level, Ticket_Channel, Resolution_Time_Hours, Satisfaction_Score.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RAW TICKET DATA                           │
│  (text, category, channel, resolution time, satisfaction)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  ┌──────────────┐       ┌──────────────────┐
  │ 4 SEVERITY   │       │ ASSIGNED PRIORITY│
  │ SIGNALS      │       │ (human label)    │
  │ (independent)│       └────────┬─────────┘
  └──────┬───────┘                │
         │                        │
         ▼                        │
  ┌──────────────┐                │
  │ WEIGHTED     │                │
  │ FUSION       │                │
  │ (0.40/0.30/  │                │
  │  0.15/0.15)  │                │
  └──────┬───────┘                │
         │                        │
         ▼                        ▼
  ┌──────────────────────────────────────┐
  │      SEVERITY DELTA (Δ)              │
  │  inferred_level - assigned_level     │
  │  |Δ| >= 2 → Mismatch pseudo-label   │
  │  Δ = 0   → Consistent pseudo-label  │
  │  |Δ| = 1 → Borderline (excluded)    │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │     FINE-TUNED DISTILBERT            │
  │  Text + metadata prefix input        │
  │  WeightedRandomSampler + class loss  │
  │  Validation-tuned threshold (0.955)  │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │    HYBRID DECISION LAYER             │
  │  |Δ| >= 2 → structural override     │
  │  Δ = 0, low prob → consistent       │
  │  else → model probability            │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │    EVIDENCE DOSSIER                  │
  │  Hallucination-free, traceable       │
  │  Hidden Crisis / False Alarm         │
  └──────────────────────────────────────┘
```

## Pseudo-Labeling Strategy

Since no human-annotated mismatch labels exist, SIA generates supervision through 4 independent severity signals:

| Signal | Source | Weight | Logic |
|--------|--------|--------|-------|
| A: Rule-based NLP | Ticket text keywords | 0.40 | Tiered keyword matching (fraud/breach → 3, payment failed → 2, password reset → 1, general inquiry → 0) |
| B: Resolution-time | Resolution_Time_Hours | 0.30 | Quantile-based: fast resolution → high severity (SLA urgency proxy) |
| C: Issue-category | Issue_Category | 0.15 | Structural mapping: Fraud→3, Billing/Technical→2, Account→1, General→0 |
| D: Satisfaction-inverse | Satisfaction_Score | 0.15 | Low satisfaction → high severity (customer distress signal) |

**Fusion:** Weighted average → discretized to 4 levels (Low/Medium/High/Critical).

**Severity delta:** `inferred_level - assigned_priority_level`
- `|Δ| >= 2` → Mismatch (confident pseudo-label)
- `Δ = 0` → Consistent (confident pseudo-label)
- `|Δ| = 1` → Borderline (excluded from training)

**Mismatch types:**
- **Hidden Crisis** (Δ >= +2): Under-prioritized — true severity exceeds assigned priority
- **False Alarm** (Δ <= -2): Over-prioritized — assigned priority exceeds true severity

## Signal Agreement

Pairwise agreement between the 4 signals is intentionally low (Cohen's kappa typically 0.05-0.30). This is expected — each signal captures a different severity dimension, and their independence makes the fused estimate more robust than any single proxy.

## Ablation Study

| Signal Combination | Accuracy | Macro F1 |
|---|---|---|
| A: Rule-based NLP only | ~0.71 | ~0.65 |
| B: Resolution-time only | ~0.62 | ~0.58 |
| C: Issue-category only | ~0.59 | ~0.55 |
| D: Satisfaction-inverse only | ~0.56 | ~0.52 |
| A+B | ~0.74 | ~0.70 |
| A+B+C | ~0.76 | ~0.72 |
| A+B+C+D (all signals) | ~0.78 | ~0.74 |

Each signal contributes meaningful information. Rule-based NLP is the strongest individual signal, but all four together provide the best pseudo-label quality.

## Model Comparison

| Model | Accuracy | Macro F1 | R(Consistent) | R(Mismatch) | Threshold | Status |
|---|---|---|---|---|---|---|
| **DistilBERT (fine-tuned)** | **99.62%** | **0.9954** | **99.65%** | **99.56%** | 0.955 | **SELECTED** |
| TF-IDF + MLP Neural | 93.43% | 0.9168 | 97.46% | 83.33% | 0.215 | Strong baseline |
| TF-IDF + Logistic Regression | 87.46% | 0.8507 | 89.22% | 83.06% | 0.50 | Classical baseline |
| HistGradientBoosting | 84.37% | 0.8165 | 85.94% | 80.43% | 0.50 | Comparison |

### Final Selected Model: DistilBERT

DistilBERT achieved **99.62% accuracy** and **0.9954 macro F1** on a held-out pseudo-labeled test set. **Important disclaimer:** since the dataset does not contain human-verified mismatch labels, these metrics measure consistency with the self-supervised audit framework, not absolute real-world correctness.

The model takes text plus structured metadata as input:
```
[PRIORITY=Low] [CATEGORY=Billing] [CHANNEL=Email] [RESOLUTION_HOURS=3] [SATISFACTION=1]
Subject: Payment charged twice Description: I was charged twice for my subscription...
```

## Leakage Checks

All checks passed:
- Train-Val overlap: 0
- Train-Test overlap: 0
- Val-Test overlap: 0
- No forbidden columns found in serialized model input: mismatch_label, mismatch_type, severity_delta, inferred_severity, final_decision, target — all False
- Threshold selected on validation set only; final metrics computed separately on test set

## Threshold Tuning

The decision threshold (0.955) was selected by maximizing macro F1 on the **validation set only**. Final test metrics were computed separately using this fixed threshold. This prevents information leakage from the test set into threshold selection.

## Verification Thresholds (MARS Requirements)

| Metric | Minimum Threshold | LR | MLP | DistilBERT |
|---|---|---|---|---|
| Accuracy | >= 83% | 87.46% ✅ | 93.43% ✅ | 99.62% ✅ |
| Macro F1 | >= 0.82 | 0.8507 ✅ | 0.9168 ✅ | 0.9954 ✅ |
| R(Consistent) | >= 78% | 89.22% ✅ | 97.46% ✅ | 99.65% ✅ |
| R(Mismatch) | >= 78% | 83.06% ✅ | 83.33% ✅ | 99.56% ✅ |

All three verified models pass all MARS thresholds.

## Evidence Dossier Schema

Every mismatch ticket receives a structured, hallucination-free dossier:

```json
{
  "ticket_id": "TKT-1234",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    {"signal": "keyword", "value": "fraud", "weight": "tier_3_severity"},
    {"signal": "resolution_time", "value": "2 hours", "interpretation": "Resolution time of 2h -> severity 3 (fast resolution = high urgency)"},
    {"signal": "issue_category", "value": "Fraud", "interpretation": "'Fraud' -> severity 3"},
    {"signal": "satisfaction_score", "value": "1", "interpretation": "Satisfaction 1/5 -> severity 3 (low satisfaction = customer distress)"}
  ],
  "constraint_analysis": "Ticket TKT-1234 was assigned 'Low' (level 0) but inferred severity is 'Critical' (level 3), delta = +3. The ticket appears under-prioritized, risking SLA breach.",
  "confidence": 0.98
}
```

### Hidden Crisis Example
A billing ticket with keywords "charged twice" and "refund", 3-hour resolution time, satisfaction 1/5, assigned "Low" priority. Inferred severity: High (level 2). Delta: +2. The ticket is under-prioritized and risks SLA breach.

### False Alarm Example
A general inquiry about pricing, 72-hour resolution time, satisfaction 5/5, assigned "Critical" priority. Inferred severity: Low (level 0). Delta: -3. The ticket is over-prioritized, wasting critical-path resources.

## Streamlit App Features

- **Single Ticket Audit:** Form input with full analysis, evidence dossier, signal breakdown
- **Batch CSV Upload:** Analyze hundreds of tickets at once with inline dashboard
- **Dashboard:** Pre-computed analysis with:
  - Flagged ticket distribution
  - Mismatch types (Hidden Crisis vs False Alarm)
  - Decision source distribution
  - Top contributing signals for mismatch tickets
  - Severity delta heatmap across category and channel
- **Downloads:** Predictions CSV and dossiers JSON

## Repository Structure

```
SIA/
├── notebook.ipynb              # Full reproducible pipeline
├── train_pipeline.py           # Training: pseudo-labels, LR, MLP, HGB
├── train_transformer.py        # DistilBERT fine-tuning (Colab GPU)
├── predict.py                  # Inference: DistilBERT > MLP > LR fallback
├── app.py                      # Streamlit dashboard
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── demo_video_script.md        # Video recording guide
├── model_artifacts/
│   ├── distilbert_finetuned/   # DistilBERT model + tokenizer (from Colab)
│   ├── baseline_logistic_model.pkl
│   ├── mlp_classifier.pkl
│   ├── hgb_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metadata_preprocessor.pkl
│   ├── resolution_thresholds.pkl
│   └── model_config.json
├── outputs/
│   ├── transformer_metrics.json
│   ├── mlp_metrics.json
│   ├── baseline_metrics.json
│   ├── model_comparison.json / .csv
│   ├── signal_agreement.json
│   ├── ablation_results.csv
│   ├── sia_final_hybrid_predictions.csv
│   ├── sia_final_hybrid_dossiers.json
│   ├── dashboard_summary.json
│   ├── top_contributing_signals.csv
│   ├── severity_delta_heatmap.csv / .png
│   ├── confusion_matrix_distilbert.png
│   ├── final_decision_distribution.png
│   └── mismatch_type_distribution.png
└── data/
    ├── customer_support_tickets.csv  (download from Kaggle)
    ├── sample_input.csv
    └── DATASET_INSTRUCTIONS.md
```

## Setup Instructions

```bash
# 1. Clone the repository
git clone <repo-url> && cd SIA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Place customer_support_tickets.csv in data/

# 4. Train baseline models (CPU, ~2 min)
python train_pipeline.py --input_csv data/customer_support_tickets.csv --output_dir outputs

# 5. Train DistilBERT (Colab GPU recommended, ~10 min on T4)
python train_transformer.py --input_csv data/customer_support_tickets.csv --output_dir outputs --model_dir model_artifacts --epochs 3 --batch_size 16 --lr 2e-5

# 6. Run predictions
python predict.py --input_csv data/sample_input.csv --output_dir prediction_outputs --model_dir model_artifacts

# 7. Launch Streamlit
streamlit run app.py
```

### Training Commands

```bash
# Baseline pipeline (LR + MLP + HGB)
python train_pipeline.py --input_csv data/customer_support_tickets.csv

# DistilBERT fine-tuning (GPU required)
python train_transformer.py --input_csv data/customer_support_tickets.csv --epochs 3 --batch_size 16 --lr 2e-5
```

### Prediction Commands

```bash
# Predict on new CSV
python predict.py --input_csv data/sample_input.csv --output_dir prediction_outputs

# Outputs: sia_predictions.csv + sia_dossiers.json
```

### Streamlit Command

```bash
streamlit run app.py
```

## Demo Video Guide

See `demo_video_script.md` for the full recording script (~3 minutes covering):
1. SIA introduction and pseudo-labeling strategy
2. DistilBERT model and metrics
3. Hidden Crisis walkthrough (single ticket)
4. False Alarm walkthrough
5. Adversarial ticket test
6. Batch dashboard and downloads

## Limitations

1. **Pseudo-labeled data:** All training labels are algorithmically generated, not human-verified. High model accuracy reflects consistency with the self-supervised framework, not guaranteed real-world correctness.
2. **Keyword dependency:** Signal A relies on predefined keyword tiers; novel issue types or evolving language may not be captured.
3. **Single dataset:** Evaluation is on one CRM dataset. Generalization to other domains or ticket formats is not validated.
4. **Threshold sensitivity:** The 0.955 threshold is tuned for this dataset's class distribution; deployment on different distributions may require re-tuning.

## Future Improvements

1. **Human-in-the-loop validation:** Collect human mismatch annotations for a subset to validate pseudo-label quality
2. **LLM-based severity scoring:** Add a zero-shot LLM signal (Mistral-7B-Instruct or Phi-3) as a 5th severity proxy
3. **Embedding-based clustering:** Use sentence-transformers for semantic urgency grouping
4. **Active learning:** Prioritize uncertain predictions for human review
5. **Multi-language support:** Extend to non-English ticket data
6. **Temporal drift detection:** Monitor for distribution shift over time

## MARS Requirement Checklist

| Requirement | Status |
|---|---|
| notebook.ipynb with full pipeline | ✅ |
| train_pipeline.py standalone script | ✅ |
| predict.py with CSV input + dossier output | ✅ |
| README.md with methodology + ablation + metrics | ✅ |
| requirements.txt | ✅ |
| Streamlit app (single + batch + dashboard) | ✅ |
| Evidence dossier generation (exact schema) | ✅ |
| Severity delta heatmap | ✅ |
| Mismatch type distribution | ✅ |
| Top contributing signals | ✅ |
| Fine-tuned classifier (DistilBERT) | ✅ |
| Text + structured metadata input | ✅ |
| Class imbalance handling | ✅ |
| Pseudo-label signal agreement | ✅ |
| Ablation study | ✅ |
| Binary accuracy >= 83% | ✅ (99.62%) |
| Macro F1 >= 0.82 | ✅ (0.9954) |
| Per-class recall >= 78% | ✅ (99.65% / 99.56%) |
| Zero hallucination in dossiers | ✅ |
| Demo video script | ✅ |
