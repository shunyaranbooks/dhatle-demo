# app.py — DHATLE Demo (complete)
# --------------------------------
# Streamlit app implementing:
# - Talent Lifecycle Engine (TLE) with gates + state memory
# - Agentic AI Layer with per-employee φ and RHML updates
# - Fairness & Resignation Risk (Fairlearn-backed metrics inside fairness.py)
# - Explainability (SHAP) with selectable Waterfall / Bar plots
# - Governance (MaturityScore) + append-only audit log

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# For SHAP visuals
import shap
import matplotlib.pyplot as plt

from dhatle.constants import STAGES, STAGE_INDEX, DEFAULT_GATES
from dhatle.engine import TalentLifecycleEngine, EmployeeState
from dhatle.agents import AgenticLayer, Governance
from dhatle.fairness import group_fairness_score, resignation_risk
from dhatle.explain import train_promotion_model, shap_explain
from dhatle.governance import maturity_score, autopilot_enabled
from dhatle.utils import append_audit, new_version_hash

# -----------------------------
# App Setup & Global Settings
# -----------------------------
st.set_page_config(page_title="DHATLE Demo", layout="wide")
st.title("DHATLE — Dynamic Human–Agentic Talent Lifecycle Ecosystem (Demo)")

warnings.filterwarnings("ignore")

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    path = os.path.join("dhatle", "data", "synthetic_employees.csv")
    df = pd.read_csv(path)
    # Light validation
    expected_cols = {
        "emp_id", "gender", "age", "dept", "tenure_months",
        "performance", "engagement", "learning_speed", "current_stage"
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        st.warning(f"Dataset missing expected columns: {missing}. Proceeding with available columns.")
    return df

df = load_data()

# -----------------------------
# Session Initialization
# -----------------------------
if "engine" not in st.session_state:
    st.session_state.engine = TalentLifecycleEngine()
if "agents" not in st.session_state:
    st.session_state.agents = AgenticLayer()
if "gov" not in st.session_state:
    st.session_state.gov = Governance()
if "model" not in st.session_state:
    model, acc = train_promotion_model(df)
    st.session_state.model = model
    st.session_state.model_acc = acc

engine: TalentLifecycleEngine = st.session_state.engine
agents: AgenticLayer = st.session_state.agents
gov: Governance = st.session_state.gov
model = st.session_state.model

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Hyperparameters / Gates")

    perf_min = st.slider("perf_min", 0.0, 1.0, float(DEFAULT_GATES["perf_min"]), 0.01)
    eng_min = st.slider("eng_min", 0.0, 1.0, float(DEFAULT_GATES["eng_min"]), 0.01)
    learn_min = st.slider("learn_min", 0.0, 1.0, float(DEFAULT_GATES["learn_min"]), 0.01)
    engine.gates.update({"perf_min": perf_min, "eng_min": eng_min, "learn_min": learn_min})

    st.markdown("---")
    st.subheader("Feedback learning rate λ")
    agents.learning_rate = st.slider("λ", 0.0, 1.0, 0.2, 0.05)

    st.subheader("Resignation risk γ weights")
    g1 = st.slider("γ1 (1-G)", 0.0, 1.0, 0.4, 0.05)
    g2 = st.slider("γ2 (1-P)", 0.0, 1.0, 0.4, 0.05)
    g3 = st.slider("γ3 (1-F)", 0.0, 1.0, 0.2, 0.05)

    st.subheader("Maturity threshold")
    mth = st.slider("Autopilot threshold", 0.5, 0.95, 0.75, 0.01)

    st.markdown("---")
    st.caption(f"Promotion model test accuracy: **{st.session_state.model_acc:.3f}**")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Overview",
    "Employee Decisions",
    "Fairness & Risk",
    "Explainability",
    "Governance & Audit"
])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tabs[0]:
    st.subheader("Workforce overview")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.histogram(df, x="current_stage", title="Employees per Stage")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="dept", color="gender",
                           barmode="group", title="Dept by Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# Tab 2: Employee Decisions
# -----------------------------
with tabs[1]:
    st.subheader("Human–Agent Decisions")

    if df.empty:
        st.warning("Dataset is empty.")
    else:
        emp_ids = df["emp_id"].tolist()
        emp_id = st.selectbox("Select employee", emp_ids, index=0)
        row = df[df.emp_id == emp_id].iloc[0]
        e = EmployeeState(
            emp_id=emp_id,
            stage=row.current_stage,
            performance=float(row.performance),
            engagement=float(row.engagement),
            learning_speed=float(row.learning_speed)
        )

        phi_row = agents.get_row(emp_id, e.stage)
        next_stage, probs = engine.decide_next(e, phi_row)

        st.markdown(f"**Current stage:** `{e.stage}` → **Proposed:** `{next_stage}`")
        st.json({k: round(v, 4) for k, v in probs.items()})

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Performance", f"{e.performance:.2f}")
        with colB:
            st.metric("Engagement", f"{e.engagement:.2f}")
        with colC:
            st.metric("Learning Speed", f"{e.learning_speed:.2f}")

        st.markdown("### Manager feedback")
        approved_choice = st.radio("Approve AI decision?", ["Approve", "Override"], horizontal=True)
        note = st.text_input("Notes / reason (optional)")

        if approved_choice == "Override":
            to_stage = st.selectbox("Override to stage", STAGES, index=STAGES.index(e.stage))
            fb = 0.0
            final_stage = to_stage
            approved_bool = False
        else:
            fb = 1.0
            final_stage = next_stage
            approved_bool = True

        if st.button("Submit decision"):
            # RHML update
            before = agents.get_row(emp_id, e.stage).get(final_stage, 0.0)
            agents.feedback_update(emp_id, final_stage, fb)
            after = agents.get_row(emp_id, e.stage).get(final_stage, 0.0)
            gov.record_feedback(emp_id, approved_bool)
            gov.track_phi_change(emp_id, after - before)

            # Append audit
            version = new_version_hash({
                "emp_id": emp_id, "from": e.stage, "to": final_stage, "probs": probs
            })
            append_audit({
                "version": version,
                "actor": "manager",
                "emp_id": emp_id,
                "action": "approve" if approved_bool else "override",
                "from_stage": e.stage,
                "to_stage": final_stage,
                "prob": round(float(probs.get(final_stage, 0.0)), 4),
                "approved": approved_bool,
                "feedback": fb,
                "notes": note or ""
            })

            # Update in df for continuity in the demo session
            df.loc[df.emp_id == emp_id, "current_stage"] = final_stage
            st.success("Decision recorded, learning updated, and audit appended.")

# -----------------------------
# Tab 3: Fairness & Risk
# -----------------------------
with tabs[2]:
    st.subheader("Group Fairness & Resignation Risk")

    feature_cols = [
        "gender", "dept", "age", "tenure_months",
        "performance", "engagement", "learning_speed"
    ]
    X = df[feature_cols]
    # Promotion propensity proxy: threshold on predicted prob
    y_pred = (st.session_state.model.predict_proba(X)[:, 1] > 0.5).astype(int)
    y_true = y_pred  # proxy for demo

    fairness = group_fairness_score(df, y_true, y_pred, sensitive_col="gender")
    st.json(fairness)

    # Map fairness rates to group-level F proxies
    group_to_F = {}
    max_rate = max(fairness["rates"].values()) if fairness["rates"] else 1.0
    for g, rate in fairness["rates"].items():
        group_to_F[g] = (rate / max_rate) if max_rate > 0 else 1.0

    gammas = (g1, g2, g3)
    df["RR"] = [
        resignation_risk(
            row.engagement,
            row.performance,
            group_to_F.get(str(row.gender), fairness["fairness"]),
            gammas
        )
        for _, row in df.iterrows()
    ]

    st.write("Top 10 resignation risks")
    st.dataframe(
        df.sort_values("RR", ascending=False).head(10)[
            ["emp_id", "gender", "dept", "performance", "engagement", "RR"]
        ],
        use_container_width=True
    )

    fig = px.histogram(df, x="RR", nbins=20, title="Distribution of Resignation Risk")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 4: Explainability (SHAP)
# -----------------------------
with tabs[3]:
    st.subheader("Explainability")
    st.caption("SHAP for promotion propensity model")

    feature_cols = [
        "gender", "dept", "age", "tenure_months",
        "performance", "engagement", "learning_speed"
    ]

    # UI controls
    idxs = st.multiselect(
        "Pick up to 5 employees to explain",
        list(range(len(df))),
        max_selections=5,
        default=[0]
    )
    plot_type = st.radio(
        "Plot type",
        ["Waterfall (matplotlib)", "Bar (Plotly)"],
        horizontal=True
    )

    if idxs:
        with st.spinner("Computing SHAP…"):
            shap_vals, base = shap_explain(st.session_state.model, df, idxs)

        # Base is log-odds; convert to probability for context
        base_prob = 1 / (1 + np.exp(-float(base)))
        st.write("Expected value (base, log-odds):", float(base))
        st.write("Baseline probability:", float(base_prob))
        st.write("SHAP array shape (rows × features):", np.array(shap_vals).shape)

        # Local explanation for the first selected employee
        X_sel = df[feature_cols].iloc[idxs]
        v = np.array(shap_vals)[0]  # shape: (n_features,)
        exp = shap.Explanation(
            values=v,
            base_values=float(base),
            data=X_sel.iloc[0].values,
            feature_names=feature_cols
        )

        if plot_type.startswith("Waterfall"):
            # Waterfall for a single instance
            fig_wf = plt.figure(figsize=(7.5, 5))
            shap.plots.waterfall(exp, show=False, max_display=12)
            st.pyplot(fig_wf)

            # Mean |SHAP| across selected instances
            mean_abs = np.abs(np.array(shap_vals)).mean(axis=0)
            order = np.argsort(mean_abs)
            fig2 = plt.figure(figsize=(7.5, 5))
            plt.barh(np.array(feature_cols)[order], mean_abs[order])
            plt.title("Mean |SHAP| across selected employees")
            plt.xlabel("|SHAP| (log-odds)")
            st.pyplot(fig2)

        else:
            # Plotly bars: local explanation and aggregate mean |SHAP|
            plot_df = pd.DataFrame({"feature": feature_cols, "shap": v})
            plot_df = plot_df.sort_values("shap")
            fig_bar = px.bar(
                plot_df, x="shap", y="feature", orientation="h",
                title="Local attributions for selected employee (log-odds)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            mean_abs = np.abs(np.array(shap_vals)).mean(axis=0)
            fig_bar2 = px.bar(
                x=mean_abs, y=feature_cols, orientation="h",
                title="Mean |SHAP| across selected employees"
            )
            st.plotly_chart(fig_bar2, use_container_width=True)

# -----------------------------
# Tab 5: Governance & Audit
# -----------------------------
with tabs[4]:
    st.subheader("Governance & Audit")

    if df.empty:
        st.warning("Dataset is empty.")
    else:
        emp_id = st.selectbox("Employee for maturity check", df.emp_id.unique(), index=0, key="gov_emp")
        score = maturity_score(gov, emp_id)
        st.metric("Maturity Score", f"{score:.2f}")
        st.write("Autopilot:", "Enabled" if autopilot_enabled(score, threshold=mth) else "Disabled")

    # Download audit log if present
    audit_path = "audit_log.csv"
    data_bytes = b""
    if os.path.exists(audit_path):
        with open(audit_path, "rb") as fh:
            data_bytes = fh.read()

    st.download_button(
        "Download audit log CSV",
        data=data_bytes,
        file_name="audit_log.csv",
        disabled=(len(data_bytes) == 0),
        help="Append-only audit CSV with version hashes for each decision."
    )

    st.caption("Audit log is append-only with version hashes. For production, wire to Postgres + pgAudit.")
