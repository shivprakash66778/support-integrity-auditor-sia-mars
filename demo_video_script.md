# SIA — Demo Video Script (~3 minutes)

## 0:00–0:20 — Introduction

> "This is the Support Integrity Auditor — SIA — built for MARS Open Projects 2026.
> SIA detects Priority Mismatch in CRM support tickets: cases where a ticket's true
> severity, inferred from text, metadata, and resolution patterns, conflicts with its
> human-assigned priority level."

**Show:** Streamlit Home page with metrics cards.

---

## 0:20–0:50 — Pseudo-Label Generation Strategy

> "The dataset has no ground-truth mismatch labels. SIA bootstraps supervision from
> 4 independent severity signals: rule-based NLP keyword detection, resolution-time
> urgency proxy, issue-category structural mapping, and satisfaction-inverse distress
> signals. These are fused with weighted averaging, and tickets where the inferred
> severity diverges from assigned priority by 2+ levels become mismatch labels.
> Borderline cases (delta = +/-1) are excluded for clean supervision."

**Show:** Notebook section on pseudo-labeling, signal table, severity delta formula.

---

## 0:50–1:30 — Model Architecture and DistilBERT

> "We trained three model families: a TF-IDF + Logistic Regression baseline at 87.5%
> accuracy, a TF-IDF + MLP neural classifier at 93.4%, and a fine-tuned DistilBERT
> transformer at 99.6% accuracy and 0.9954 macro F1 on held-out pseudo-labeled test data.
> DistilBERT was selected as the final model. It takes text plus structured metadata —
> category, channel, resolution time, satisfaction — as a formatted input prefix.
> Class imbalance is handled via WeightedRandomSampler and class-weighted loss.
> The decision threshold was tuned on validation data only."

> "Important: these metrics measure consistency with the self-supervised audit framework,
> not absolute real-world accuracy, since we have no human-verified mismatch labels."

**Show:** Model comparison table, confusion matrix, threshold tuning plot.

---

## 1:30–2:00 — Hidden Crisis Walkthrough

> "Here is a Hidden Crisis example. I'll enter a ticket about a payment failure —
> charged twice, needs an immediate refund — but assign it Low priority."

**Demo:** Single Ticket tab:
- Subject: "Payment charged twice — urgent refund needed"
- Category: Billing
- Priority: Low
- Resolution Time: 3 hours
- Satisfaction: 1/5

> "SIA flags this as a Hidden Crisis with high confidence. The evidence dossier shows
> keyword matches for 'charged twice' and 'refund' at tier-2 severity, fast resolution
> time indicating urgency, and very low satisfaction indicating customer distress.
> The severity delta is +2, confirming the ticket is under-prioritized."

**Show:** Mismatch result, evidence dossier JSON, signal breakdown.

---

## 2:00–2:20 — False Alarm Walkthrough

> "Now a False Alarm. I'll enter a general inquiry about pricing, but assign it Critical."

**Demo:** Single Ticket tab:
- Subject: "Question about pricing and plans"
- Category: General Inquiry
- Priority: Critical
- Resolution Time: 72 hours
- Satisfaction: 5/5

> "SIA flags this as a False Alarm — the ticket is over-prioritized.
> Keyword severity is 0, satisfaction is high, and resolution was slow,
> all indicating a low-severity ticket that shouldn't consume critical-path resources."

---

## 2:20–2:40 — Adversarial Test

> "For the adversarial test: a ticket that uses calm language but describes a severe issue."

**Demo:** Single Ticket tab:
- Subject: "Account access question"
- Description: "Hi, I noticed some transactions I did not make on my account. Someone appears to have gained access. Could you look into this when convenient?"
- Category: Account
- Priority: Low

> "Despite the polite tone, SIA detects keywords like 'transactions I did not make'
> and 'gained access' — consistent with unauthorized access. It correctly flags
> this as a Hidden Crisis."

---

## 2:40–3:00 — Dashboard and Deliverables

> "The batch dashboard shows mismatch distribution, mismatch types, top contributing
> signals, and a severity delta heatmap across categories and channels.
> All outputs — predictions CSV, evidence dossiers JSON, and dashboard files —
> are downloadable. The full repository includes the training pipeline, prediction
> script, Streamlit app, and a clean notebook covering the entire pipeline."

**Show:** Dashboard tab with charts, download buttons, file structure.

> "Thank you — this has been the Support Integrity Auditor for MARS Open Projects 2026."

---

## Recording Notes

- Use screen recording software (OBS or built-in)
- Record at 1080p minimum
- Keep the Streamlit app running: `streamlit run app.py`
- Upload to YouTube or Google Drive and share the link
