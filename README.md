# Support Integrity Auditor (SIA)

**MARS Open Projects 2026 — Problem Statement 1**  
**Domain:** Artificial Intelligence / Machine Learning / NLP / CRM Systems  

---

## Important Links

| Resource | Link |
|---|---|
| GitHub Repository | [support-integrity-auditor](https://github.com/shivprakash66778/support-integrity-auditor-sia-mars) |
| Streamlit App | [ SIA App](https://3ut8nqtpaanvre6wpsgust.streamlit.app/) |
| Demo Video | [Drive Link](https://drive.google.com/drive/folders/1SL-iCYx_ueRuq31Mz8tSH39UuaBtFEnP?usp=sharing) |
| DistilBERT Model Artifacts | [Drive Link](https://drive.google.com/drive/folders/1SL-iCYx_ueRuq31Mz8tSH39UuaBtFEnP?usp=sharing) |

---

## Problem Overview

In enterprise CRM ecosystems, manual support-ticket triage can suffer from agent fatigue, keyword anchoring, priority inflation, and priority deflation. A critical issue may be incorrectly marked as Low, while a trivial general inquiry may be escalated as Critical. Such mismatches can affect SLA compliance, resource allocation, and customer experience.

**Support Integrity Auditor (SIA)** is a semantics-driven, evidence-grounded automated auditor that detects **Priority Mismatch** in customer support tickets. A priority mismatch occurs when the objective characteristics of a ticket, such as ticket text, issue category, channel, resolution time, and satisfaction score, conflict with the human-assigned priority.

The key challenge is that the dataset does not contain pre-annotated mismatch labels. Therefore, SIA first creates its own self-supervised pseudo-labels and then trains supervised models on those labels.

---

## Dataset

**Dataset:** Customer Support Tickets — CRM Dataset  
**Source:** Kaggle  
**Expected file path:** `data/customer_support_tickets.csv`

The dataset contains 20,000 customer support tickets with text, priority labels, issue metadata, resolution time, and satisfaction scores.

### Key Columns Used

| Column | Role |
|---|---|
| `Ticket_Subject` | Short summary of the issue |
| `Ticket_Description` | Full natural language ticket description |
| `Priority_Level` | Human-assigned priority: Low, Medium, High, Critical |
| `Issue_Category` | Ticket type/category |
| `Ticket_Channel` | Intake channel such as Email, Chat, or Web Form |
| `Resolution_Time_Hours` | Time taken to resolve the ticket |
| `Satisfaction_Score` | Customer satisfaction score |

---

## Project Architecture

```text
Raw CRM Tickets
     |
     v
Text Cleaning + Metadata Processing
     |
     v
Independent Severity Signals
     |---- Rule-based NLP Severity
     |---- Resolution-time Severity
     |---- Issue-category Severity
     |---- Satisfaction-inverse Severity
     |
     v
Weighted Signal Fusion
     |
     v
Inferred Severity Level
     |
     v
Severity Delta = Inferred Severity - Assigned Priority
     |
     |---- Delta >= +2  -> Hidden Crisis
     |---- Delta <= -2  -> False Alarm
     |---- Delta = 0    -> Consistent
     |---- |Delta| = 1  -> Borderline / excluded from training
     |
     v
Pseudo-labeled Training Data
     |
     v
Model Training
     |---- TF-IDF + Logistic Regression
     |---- TF-IDF + MLP
     |---- HistGradientBoosting
     |---- Fine-tuned DistilBERT
     |
     v
Final Prediction + Evidence Dossier
     |
     v
Streamlit Dashboard + Batch Inference
```

---

## Pseudo-Labeling Strategy

Since the dataset does not provide human-verified mismatch labels, SIA generates pseudo-labels using independent severity signals. The goal is to infer the likely true severity of a ticket without directly depending on the assigned priority.

### Severity Signals

| Signal | Source | Weight | Logic |
|---|---|---:|---|
| Rule-based NLP Severity | Ticket subject and description | 0.40 | Detects urgency keywords such as fraud, breach, payment failed, refund, access denied, general inquiry |
| Resolution-time Severity | `Resolution_Time_Hours` | 0.30 | Faster resolution is treated as a proxy for urgency/SLA pressure |
| Issue-category Severity | `Issue_Category` | 0.15 | Fraud is mapped high; Billing/Technical medium-high; Account medium; General Inquiry low |
| Satisfaction-inverse Severity | `Satisfaction_Score` | 0.15 | Lower satisfaction implies higher customer distress |

The weighted severity score is discretized into four levels:

```text
0 -> Low
1 -> Medium
2 -> High
3 -> Critical
```

### Severity Delta

```text
severity_delta = inferred_severity_level - assigned_priority_level
```

| Condition | Label | Interpretation |
|---|---|---|
| `severity_delta >= +2` | Mismatch | Hidden Crisis: under-prioritized ticket |
| `severity_delta <= -2` | Mismatch | False Alarm: over-prioritized ticket |
| `severity_delta = 0` | Consistent | Assigned priority matches inferred severity |
| `abs(severity_delta) = 1` | Borderline | Excluded from confident training labels |

Final evidence dossiers are generated only for directional mismatches:

```text
severity_delta > 0  -> Hidden Crisis
severity_delta < 0  -> False Alarm
severity_delta = 0  -> excluded from final evidence dossier JSON
```

This avoids ambiguous dossier cases where the model flags a mismatch but the structural severity delta has no direction.

---

## Signal Agreement and Ablation

Pairwise agreement between severity signals is intentionally not perfect because each signal captures a different aspect of severity. Rule-based keywords capture language urgency, resolution time captures SLA pressure, issue category captures structural risk, and satisfaction score captures customer distress.

### Ablation Summary

| Signal Combination | Accuracy | Macro F1 | Observation |
|---|---:|---:|---|
| Rule-based NLP only | ~0.71 | ~0.65 | Strongest individual signal |
| Resolution-time only | ~0.62 | ~0.58 | Useful operational proxy |
| Issue-category only | ~0.59 | ~0.55 | Captures structural risk |
| Satisfaction-inverse only | ~0.56 | ~0.52 | Captures customer distress |
| Rule + Resolution | ~0.74 | ~0.70 | Improves over single-signal setup |
| Rule + Resolution + Category | ~0.76 | ~0.72 | Better semantic and metadata fusion |
| All four signals | ~0.78 | ~0.74 | Best pseudo-label quality |

---

## Models Tried

The project compares classical ML, neural baseline, and transformer-based approaches.

### 1. TF-IDF + Logistic Regression

A classical baseline using TF-IDF text features and structured metadata. It provides an interpretable and lightweight benchmark.

### 2. TF-IDF + MLP Neural Classifier

A neural baseline trained on TF-IDF text features and metadata-derived signals. It improves substantially over Logistic Regression.

### 3. HistGradientBoosting

A tree-based comparison model trained on engineered structured features.

### 4. Fine-tuned DistilBERT

The final selected model. DistilBERT is fine-tuned on pseudo-labeled tickets using text plus structured metadata serialized into the input prompt.

Example model input:

```text
[PRIORITY=Low] [CATEGORY=Billing] [CHANNEL=Email] [RESOLUTION_HOURS=3] [SATISFACTION=1]
Subject: Payment charged twice
Description: I was charged twice for my subscription and need a refund urgently.
```

---

## Final Model Comparison

| Model | Accuracy | Macro F1 | Recall Consistent | Recall Mismatch | Threshold | Status |
|---|---:|---:|---:|---:|---:|---|
| **Fine-tuned DistilBERT** | **99.62%** | **0.9954** | **99.65%** | **99.56%** | 0.955 | **Selected final model** |
| TF-IDF + MLP | 93.43% | 0.9168 | 97.46% | 83.33% | 0.215 | Strong neural baseline |
| TF-IDF + Logistic Regression | 87.46% | 0.8507 | 89.22% | 83.06% | 0.50 | Classical baseline |
| HistGradientBoosting | 84.37% | 0.8165 | 85.94% | 80.43% | 0.50 | Comparison model |

---

## Important Evaluation Disclaimer

DistilBERT achieved **99.62% accuracy** and **0.9954 macro F1** on a held-out pseudo-labeled test set.

Since the dataset does not contain human-verified mismatch labels, these metrics measure **consistency with the self-supervised audit framework**, not absolute real-world correctness. The model should therefore be interpreted as a strong automated audit assistant rather than a replacement for human review in high-stakes support operations.

---

## Leakage and Validation Checks

The following checks were performed to verify that the high DistilBERT score was not caused by obvious leakage:

| Check | Result |
|---|---|
| Train-Val overlap | 0 |
| Train-Test overlap | 0 |
| Val-Test overlap | 0 |
| `mismatch_label` in serialized input | False |
| `mismatch_type` in serialized input | False |
| `severity_delta` in serialized input | False |
| `inferred_severity` in serialized input | False |
| `final_decision` in serialized input | False |
| `target` in serialized input | False |

The decision threshold of **0.955** was selected on the validation set only. Final test metrics were computed separately using this fixed validation-selected threshold.

---

## MARS Verification Thresholds

| Metric | Required Minimum | Logistic Regression | MLP | DistilBERT |
|---|---:|---:|---:|---:|
| Accuracy | >= 83% | 87.46% | 93.43% | 99.62% |
| Macro F1 | >= 0.82 | 0.8507 | 0.9168 | 0.9954 |
| Recall Consistent | >= 78% | 89.22% | 97.46% | 99.65% |
| Recall Mismatch | >= 78% | 83.06% | 83.33% | 99.56% |

All major models pass the required MARS thresholds. DistilBERT is selected as the final model.

---

## Evidence Dossier

For every final directional mismatch, SIA generates a structured evidence dossier.

### Schema

```json
{
  "ticket_id": "TKT-1234",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "fraud",
      "weight": "tier_3_severity"
    },
    {
      "signal": "resolution_time",
      "value": "2 hours",
      "interpretation": "Resolution time of 2h implies high SLA urgency"
    }
  ],
  "constraint_analysis": "Ticket TKT-1234 was assigned Low priority, but the inferred severity is Critical. The positive severity delta indicates under-prioritization and potential SLA risk.",
  "confidence": 0.98
}
```

### Dossier Grounding Rules

- Every evidence item is traceable to an input field or computed severity signal.
- No fabricated or unverifiable evidence is included.
- Final dossiers contain only:
  - Hidden Crisis
  - False Alarm
- `severity_delta = 0` cases are excluded from the final formal dossier JSON.

---

## Example Cases

### Hidden Crisis

A ticket is assigned **Low** priority but contains billing failure language, low satisfaction, and fast resolution time.

```text
Subject: Payment failed twice
Description: My payment failed twice and I cannot access my account.
Assigned Priority: Low
Issue Category: Billing
Resolution Time: 5 hours
Satisfaction Score: 1
```

Expected result:

```text
Final Decision: Mismatch
Mismatch Type: Hidden Crisis
Reason: Inferred severity is higher than assigned priority
```

### False Alarm

A ticket is assigned **Critical** priority but is only a general inquiry.

```text
Subject: Office hours
Description: I want to know where your office hours are listed.
Assigned Priority: Critical
Issue Category: General Inquiry
Resolution Time: 80 hours
Satisfaction Score: 5
```

Expected result:

```text
Final Decision: Mismatch
Mismatch Type: False Alarm
Reason: Assigned priority is higher than inferred severity
```

---

## Streamlit App Features

The Streamlit app provides an interactive interface for support-ticket auditing.

### Features

- Single-ticket audit form
- Batch CSV upload
- Binary judgment: Consistent or Mismatch
- Mismatch type: Hidden Crisis or False Alarm
- Assigned priority and inferred severity display
- Severity delta explanation
- Model confidence score
- Full evidence dossier
- Batch prediction download
- Dossier JSON download
- Priority mismatch dashboard

### Dashboard Includes

- Distribution of final decisions
- Hidden Crisis vs False Alarm distribution
- Decision source distribution
- Top contributing signals
- Severity delta heatmap across issue category and ticket channel

---

## Repository Structure

```text
SIA/
├── notebook.ipynb
├── train_pipeline.py
├── train_transformer.py
├── predict.py
├── app.py
├── README.md
├── requirements.txt
├── demo_video_script.md
├── model_artifacts/
│   ├── distilbert_finetuned/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── baseline_logistic_model.pkl
│   ├── tfidf_mlp_model.pkl
│   ├── hgb_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metadata_preprocessor.pkl
│   ├── resolution_thresholds.pkl
│   ├── mlp_resolution_thresholds.pkl
│   ├── mlp_model_config.json
│   └── model_config.json
├── outputs/
│   ├── transformer_experiment_metrics.json
│   ├── mlp_metrics.json
│   ├── model_comparison_transformer_experiment.csv
│   ├── model_comparison_mlp.csv
│   ├── signal_agreement.json
│   ├── ablation_results.csv
│   ├── sia_final_hybrid_predictions.csv
│   ├── sia_final_hybrid_dossiers.json
│   ├── dashboard_summary.json
│   ├── top_contributing_signals.csv
│   ├── severity_delta_heatmap.csv
│   ├── transformer_confusion_matrix.png
│   ├── final_decision_distribution.png
│   └── mismatch_type_distribution.png
└── data/
    ├── sample_input.csv
    └── DATASET_INSTRUCTIONS.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone paste-github-repo-link-here
cd support-integrity-auditor-sia
```

### 2. Create Environment

```bash
python -m venv venv
```

Activate environment:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

Download the Kaggle dataset and place it here:

```text
data/customer_support_tickets.csv
```

For quick testing, use:

```text
data/sample_input.csv
```

### 5. Add DistilBERT Artifacts

If the DistilBERT model folder is not included in GitHub due to file-size limits, download it from the model artifact link above and place it here:

```text
model_artifacts/distilbert_finetuned/
```

The folder should contain:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

### 6. Run Prediction

```bash
python predict.py --input_csv data/sample_input.csv --output_dir prediction_outputs --model_dir model_artifacts
```

Expected output:

```text
DistilBERT loaded from model_artifacts/distilbert_finetuned
SIA Prediction — Model: DISTILBERT
```

### 7. Run Streamlit App

```bash
streamlit run app.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run app.py
```

---

## Training Commands

### Train Classical and MLP Models

```bash
python train_pipeline.py --input_csv data/customer_support_tickets.csv --output_dir outputs
```

### Train DistilBERT

GPU is recommended.

```bash
python train_transformer.py --input_csv data/customer_support_tickets.csv --output_dir outputs --model_dir model_artifacts --epochs 3 --batch_size 16 --lr 2e-5
```

---

## Prediction Commands

```bash
python predict.py --input_csv data/sample_input.csv --output_dir prediction_outputs --model_dir model_artifacts
```

Outputs:

```text
prediction_outputs/sia_predictions.csv
prediction_outputs/sia_dossiers.json
```

---

## Demo Video Guide

See `demo_video_script.md` for the full 3-minute recording script.

The demo should show:

1. Brief SIA introduction
2. Pseudo-labeling strategy
3. Four severity signals
4. Final DistilBERT model and metrics
5. Hidden Crisis example
6. False Alarm example
7. Adversarial input example
8. Batch upload and dashboard
9. Downloadable predictions and dossiers

---

## Limitations

1. The project uses pseudo-labels, not human-verified mismatch labels.
2. High DistilBERT performance measures consistency with the pseudo-labeling framework.
3. Rule-based severity signal may miss new issue types or unusual language.
4. Evaluation is based on one CRM dataset.
5. Deployment on different organizations may require threshold re-tuning.
6. DistilBERT artifacts may be large for GitHub and may need external hosting.

---

## Future Improvements

- Human-in-the-loop validation using expert-labeled mismatch samples
- Add open-source LLM zero-shot severity scoring as an additional signal
- Add sentence-transformer embedding clustering for semantic urgency grouping
- Add active learning for uncertain audit cases
- Add multilingual ticket support
- Add model drift monitoring for real CRM deployment
- Improve explainability with token-level highlights for DistilBERT decisions

---

## MARS Requirement Checklist

| Requirement | Status |
|---|---|
| Full reproducible `notebook.ipynb` | Done |
| `train_pipeline.py` standalone training script | Done |
| `train_transformer.py` for DistilBERT fine-tuning | Done |
| `predict.py` with CSV input | Done |
| Prediction CSV output | Done |
| Evidence dossier JSON output | Done |
| Professional `README.md` | Done |
| `requirements.txt` | Done |
| Streamlit single-ticket input | Done |
| Streamlit batch CSV upload | Done |
| Dashboard distribution of flagged tickets | Done |
| Mismatch type distribution | Done |
| Top contributing signals | Done |
| Severity delta heatmap | Done |
| Pseudo-label generation | Done |
| At least two independent severity signals | Done, four signals used |
| Fine-tuned model | Done, DistilBERT |
| Text + metadata input | Done |
| Class imbalance handling | Done |
| Signal agreement | Done |
| Ablation study | Done |
| Accuracy >= 83% | Done, 99.62% |
| Macro F1 >= 0.82 | Done, 0.9954 |
| Per-class recall >= 78% | Done |
| Hallucination-free evidence dossier | Done |
| Demo video script | Done |
| Hosted Streamlit URL | Paste link above |
| Demo video URL | Paste link above |

---

## Author

**Shiv Prakash Vishwari**  
Indian Institute of Technology Roorkee  
MARS Open Projects 2026  
