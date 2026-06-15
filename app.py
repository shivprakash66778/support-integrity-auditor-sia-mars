#!/usr/bin/env python3
"""SIA — Streamlit Dashboard (supports DistilBERT / MLP / LR)"""
import io, json, os, sys
import joblib, numpy as np, pandas as pd, streamlit as st

try:    _ROOT = os.path.dirname(os.path.abspath(__file__))
except: _ROOT = os.getcwd()
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from predict import load_artifacts, predict_batch
from train_pipeline import generate_dossier, PRIORITY_MAP, SEVERITY_LABELS, ensure_dirs

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🔍", layout="wide")

def _find(name):
    for c in [os.path.join(_ROOT, name), os.path.join(os.getcwd(), name),
              f"/content/SIA/{name}", f"/content/{name}"]:
        if os.path.isdir(c): return c
    return os.path.join(_ROOT, name)

MODEL_DIR, OUTPUTS_DIR = _find("model_artifacts"), _find("outputs")
REQUIRED_COLS = ["Ticket_ID","Ticket_Subject","Ticket_Description","Issue_Category",
                 "Priority_Level","Ticket_Channel","Resolution_Time_Hours","Satisfaction_Score"]

@st.cache_resource
def _load():
    try: return load_artifacts(MODEL_DIR)
    except Exception as e:
        st.error(f"Cannot load model: {e}")
        st.info("Run `python train_pipeline.py --input_csv data/customer_support_tickets.csv` first.")
        st.stop()

lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bm, bt, selected, bt_thresh = _load()

MODEL_DISPLAY = {
    "distilbert": "DistilBERT (fine-tuned)",
    "mlp": "TF-IDF + MLP Neural",
    "lr": "TF-IDF + Logistic Regression",
    "hgb": "HistGradientBoosting",
}

st.sidebar.title("🔍 SIA Navigator")
page = st.sidebar.radio("", ["🏠 Home","📝 Single Ticket","📤 Batch Upload","📊 Dashboard","ℹ️ About"])
st.sidebar.markdown(f"---\n**Model:** `{MODEL_DISPLAY.get(selected, selected)}`")
st.sidebar.caption("MARS Open Projects 2026")

# ── HOME ──
if page == "🏠 Home":
    st.title("🔍 Support Integrity Auditor (SIA)")
    st.markdown("""
**Detecting Priority Mismatch in CRM Support Tickets**

SIA infers true ticket severity from 4 independent signals, compares against assigned priority,
and flags **Hidden Crises** (under-prioritized) and **False Alarms** (over-prioritized).

| Signal | Source | Weight |
|--------|--------|--------|
| Rule-based NLP | Ticket text keywords | 0.40 |
| Resolution-time proxy | Resolution hours | 0.30 |
| Issue-category mapping | Category field | 0.15 |
| Satisfaction-inverse | Satisfaction score | 0.15 |
""")
    sp = os.path.join(OUTPUTS_DIR, "dashboard_summary.json")
    if os.path.exists(sp):
        with open(sp) as f: s = json.load(f)
        mm = s.get("model_metrics", {})
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{mm.get('accuracy',0):.1%}")
        c2.metric("Macro F1", f"{mm.get('macro_f1',0):.3f}")
        c3.metric("Mismatch Rate", f"{s.get('mismatch_rate',0):.1%}")
        c4.metric("Total Tickets", f"{s.get('total_tickets',0):,}")

    # Show DistilBERT metrics if available
    tp = os.path.join(OUTPUTS_DIR, "transformer_metrics.json")
    if os.path.exists(tp):
        with open(tp) as f: tm = json.load(f)
        tuned = tm.get("tuned_threshold_metrics", {})
        if tuned:
            st.markdown("---")
            st.subheader("DistilBERT Fine-Tuned Performance")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{tuned.get('accuracy',0):.4f}")
            c2.metric("Macro F1", f"{tuned.get('macro_f1',0):.4f}")
            c3.metric("R(Consistent)", f"{tuned.get('recall_consistent',0):.4f}")
            c4.metric("R(Mismatch)", f"{tuned.get('recall_mismatch',0):.4f}")
            st.caption("Evaluated on held-out pseudo-labeled test data. These metrics measure consistency "
                      "with the self-supervised audit framework, not absolute real-world correctness.")

# ── SINGLE TICKET ──
elif page == "📝 Single Ticket":
    st.title("📝 Analyze a Single Ticket")
    with st.form("ticket"):
        c1, c2 = st.columns(2)
        with c1:
            tid  = st.text_input("Ticket ID", "TKT-CUSTOM-001")
            name = st.text_input("Customer Name", "Test User")
            email = st.text_input("Email", "test@example.com")
            subj = st.text_input("Subject", "Payment failed - Urgent")
            cat  = st.selectbox("Category", ["Technical","Billing","Account","General Inquiry","Fraud"])
        with c2:
            pri  = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
            chan = st.selectbox("Channel", ["Chat","Email","Web Form","Phone","Social Media"])
            hrs  = st.number_input("Resolution Time (h)", 1, 200, 5)
            sat  = st.slider("Satisfaction", 1, 5, 3)
        desc = st.text_area("Description", "My payment failed twice and I was charged. I need an immediate refund.", height=100)
        go = st.form_submit_button("🔍 Analyze", use_container_width=True)
    if go:
        row_df = pd.DataFrame([{"Ticket_ID":tid,"Customer_Name":name,"Customer_Email":email,
            "Ticket_Subject":subj,"Ticket_Description":desc,"Issue_Category":cat,
            "Priority_Level":pri,"Ticket_Channel":chan,"Submission_Date":"2026-01-01",
            "Resolution_Time_Hours":hrs,"Assigned_Agent":"SIA-Auto","Satisfaction_Score":sat,
            "Product_Purchased":"N/A","Ticket_Type":"N/A"}])
        with st.spinner("Analyzing..."):
            res = predict_batch(row_df, lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bm, bt, bt_thresh)
        r = res.iloc[0]
        st.markdown("---")
        if r["final_decision"]=="Mismatch":
            st.error(f"⚠️ **MISMATCH** — {r['mismatch_type']} (confidence {r['final_confidence']:.0%})")
        else:
            st.success(f"✅ **CONSISTENT** — Priority aligns with inferred severity (confidence {r['final_confidence']:.0%})")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Assigned", r["Priority_Level"]); m2.metric("Inferred", r["inferred_severity"])
        m3.metric("Severity Delta", f"{int(r['severity_delta']):+d}"); m4.metric("P(mismatch)", f"{r['mismatch_probability']:.1%}")
        st.caption(f"Decision source: `{r['decision_source']}` | Model: `{selected}`")
        if r["final_decision"]=="Mismatch":
            st.subheader("📋 Evidence Dossier")
            st.json(generate_dossier(r))
        st.subheader("🔬 Signal Breakdown")
        st.dataframe(pd.DataFrame({
            "Signal":["Rule NLP","Resolution Time","Category","Satisfaction"],
            "Severity Level":[int(r["rule_severity"]),int(r["resolution_severity"]),int(r["category_severity"]),int(r["satisfaction_severity"])],
            "Weight":[0.40,0.30,0.15,0.15],
        }), use_container_width=True, hide_index=True)

# ── BATCH UPLOAD ──
elif page == "📤 Batch Upload":
    st.title("📤 Batch CSV Upload")
    st.info(f"Required columns: `{'`, `'.join(REQUIRED_COLS)}`")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_in = pd.read_csv(up)
        missing = [c for c in REQUIRED_COLS if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}"); st.stop()
        st.write(f"**{len(df_in)} tickets loaded**")
        if st.button("🔍 Analyze All", use_container_width=True):
            with st.spinner(f"Analyzing {len(df_in)} tickets..."):
                df_r = predict_batch(df_in, lr, tfidf, meta, thresh, mlp, hgb, ohe, scaler, bm, bt, bt_thresh)
            mm = (df_r["final_decision"]=="Mismatch").sum()
            st.markdown("---")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",len(df_r)); c2.metric("Mismatches",mm)
            c3.metric("Rate",f"{mm/len(df_r):.1%}")
            hc = (df_r["mismatch_type"]=="Hidden Crisis").sum()
            fa = (df_r["mismatch_type"]=="False Alarm").sum()
            c4.metric("Hidden Crisis / False Alarm", f"{hc} / {fa}")

            show = [c for c in ["Ticket_ID","Ticket_Subject","Priority_Level","inferred_severity",
                "severity_delta","mismatch_type","mismatch_probability","final_decision","final_confidence"] if c in df_r.columns]
            st.dataframe(df_r[show], use_container_width=True, height=400)

            # Inline dashboard for batch
            st.markdown("---")
            st.subheader("Batch Analysis")
            t1,t2 = st.columns(2)
            with t1:
                st.markdown("**Decision Distribution**")
                st.bar_chart(df_r["final_decision"].value_counts())
            with t2:
                st.markdown("**Mismatch Types**")
                st.bar_chart(df_r["mismatch_type"].value_counts())

            # Top signals
            mm_df = df_r[df_r["final_decision"]=="Mismatch"]
            sig_cols = [c for c in ["rule_severity","resolution_severity","category_severity","satisfaction_severity"] if c in mm_df.columns]
            if len(mm_df) and sig_cols:
                st.markdown("**Top Contributing Signals (Mismatch tickets)**")
                sig_means = mm_df[sig_cols].mean().rename({
                    "rule_severity":"Rule NLP","resolution_severity":"Resolution",
                    "category_severity":"Category","satisfaction_severity":"Satisfaction"})
                st.bar_chart(sig_means)

            # Delta heatmap
            if "severity_delta" in df_r.columns and "Issue_Category" in df_r.columns and "Ticket_Channel" in df_r.columns:
                st.markdown("**Severity Delta Heatmap: Category x Channel**")
                piv = df_r.pivot_table(values="severity_delta", index="Issue_Category", columns="Ticket_Channel", aggfunc="mean").round(2)
                st.dataframe(piv.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)

            # Downloads
            cl1,cl2 = st.columns(2)
            with cl1:
                st.download_button("⬇️ Predictions CSV", df_r[show].to_csv(index=False), "sia_predictions.csv","text/csv", use_container_width=True)
            with cl2:
                dos = [generate_dossier(row) for _,row in mm_df.iterrows()]
                st.download_button("⬇️ Dossiers JSON", json.dumps(dos,indent=2), "sia_dossiers.json","application/json", use_container_width=True)

# ── DASHBOARD ──
elif page == "📊 Dashboard":
    st.title("📊 Priority Mismatch Dashboard")
    pp = os.path.join(OUTPUTS_DIR, "sia_final_hybrid_predictions.csv")
    if not os.path.exists(pp):
        st.warning(f"No data at `{pp}`. Run train_pipeline.py first."); st.stop()
    df = pd.read_csv(pp)
    mm = (df["final_decision"]=="Mismatch").sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total",f"{len(df):,}"); c2.metric("Mismatch",f"{mm:,}")
    c3.metric("Hidden Crisis",f"{(df['mismatch_type']=='Hidden Crisis').sum():,}")
    c4.metric("False Alarm",f"{(df['mismatch_type']=='False Alarm').sum():,}")
    c5.metric("Borderline",f"{(df['mismatch_type']=='Borderline').sum():,}")

    st.markdown("---")
    t1,t2 = st.columns(2)
    with t1: st.subheader("Decision Distribution"); st.bar_chart(df["final_decision"].value_counts())
    with t2: st.subheader("Mismatch Types"); st.bar_chart(df["mismatch_type"].value_counts())
    t3,t4 = st.columns(2)
    with t3:
        if "decision_source" in df.columns:
            st.subheader("Decision Source"); st.bar_chart(df["decision_source"].value_counts())
    with t4: st.subheader("Priority Distribution"); st.bar_chart(df["Priority_Level"].value_counts())

    st.markdown("---")
    st.subheader("Severity Delta Heatmap: Category x Channel")
    if "severity_delta" in df.columns:
        piv = df.pivot_table(values="severity_delta", index="Issue_Category", columns="Ticket_Channel", aggfunc="mean").round(2)
        st.dataframe(piv.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)

    st.subheader("Top Contributing Signals (Mismatch tickets)")
    mm_df = df[df["final_decision"]=="Mismatch"]
    sig_cols = [c for c in ["rule_severity","resolution_severity","category_severity","satisfaction_severity"] if c in mm_df.columns]
    if len(mm_df) and sig_cols:
        sig_means = mm_df[sig_cols].mean()
        st.bar_chart(sig_means.rename({"rule_severity":"Rule NLP","resolution_severity":"Resolution",
            "category_severity":"Category","satisfaction_severity":"Satisfaction"}))

    st.subheader("Mismatch Rate by Category")
    if "Issue_Category" in df.columns:
        gp = df.groupby("Issue_Category").agg(n=("final_decision","count"),mm=("final_decision",lambda x:(x=="Mismatch").sum()))
        gp["rate_%"] = (gp["mm"]/gp["n"]*100).round(1)
        st.dataframe(gp, use_container_width=True)

    # Model metrics section
    st.markdown("---")
    st.subheader("Model Performance")

    tp = os.path.join(OUTPUTS_DIR, "transformer_metrics.json")
    if os.path.exists(tp):
        with open(tp) as f: tm = json.load(f)
        tuned = tm.get("tuned_threshold_metrics", {})
        if tuned:
            st.markdown("**DistilBERT (Fine-Tuned) — Selected Model**")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Accuracy",f"{tuned.get('accuracy',0):.4f}")
            mc2.metric("Macro F1",f"{tuned.get('macro_f1',0):.4f}")
            mc3.metric("R(Consistent)",f"{tuned.get('recall_consistent',0):.4f}")
            mc4.metric("R(Mismatch)",f"{tuned.get('recall_mismatch',0):.4f}")
            st.caption("Evaluated on held-out pseudo-labeled test data.")

    mp = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f: mt = json.load(f)
        m = mt.get("baseline", mt)
        st.markdown("**Logistic Regression Baseline**")
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Accuracy",f"{m['accuracy']:.4f}"); mc2.metric("F1",f"{m['macro_f1']:.4f}")
        mc3.metric("R(Cons)",f"{m['recall_consistent']:.4f}"); mc4.metric("R(Mismatch)",f"{m['recall_mismatch']:.4f}")

    cp = os.path.join(OUTPUTS_DIR, "model_comparison.json")
    if os.path.exists(cp):
        st.subheader("Full Model Comparison")
        with open(cp) as f: comp = json.load(f)
        st.dataframe(pd.DataFrame(comp).T, use_container_width=True)

    st.markdown("---")
    cl1,cl2 = st.columns(2)
    with cl1:
        st.download_button("⬇️ Full Predictions", df.to_csv(index=False), "sia_full_predictions.csv","text/csv", use_container_width=True)
    with cl2:
        dp = os.path.join(OUTPUTS_DIR, "sia_final_hybrid_dossiers.json")
        if os.path.exists(dp):
            with open(dp) as f: dd = f.read()
            st.download_button("⬇️ Dossiers JSON", dd, "sia_dossiers.json","application/json", use_container_width=True)

# ── ABOUT ──
elif page == "ℹ️ About":
    st.title("ℹ️ About SIA")
    st.markdown("""
### Support Integrity Auditor (SIA)
**MARS Open Projects 2026 — Problem Statement 1**

SIA is a semantics-driven, evidence-grounded automated auditor that detects
**Priority Mismatch** in CRM support tickets. It addresses the fundamental
challenge that no pre-annotated mismatch labels exist — the system bootstraps
its own supervision from raw ticket data.

**Model Hierarchy:**
1. **Fine-tuned DistilBERT** — Selected final model (99.62% accuracy, 0.9954 F1 on pseudo-labeled test set)
2. **TF-IDF + MLP Neural Classifier** — Strong neural baseline (93.43% accuracy, 0.9168 F1)
3. **TF-IDF + Logistic Regression** — Classical baseline (87.46% accuracy, 0.8507 F1)
4. **HistGradientBoosting** — Comparison model

**Important Note:** DistilBERT achieved 99.62% accuracy and 0.9954 macro F1 on a held-out
pseudo-labeled test set. Since the dataset does not contain human-verified mismatch labels,
these metrics measure consistency with the self-supervised audit framework, not absolute
real-world correctness.

**Evidence Dossier:** Every flagged ticket gets a structured, hallucination-free dossier
with traceable feature evidence, constraint analysis, and calibrated confidence.
""")
