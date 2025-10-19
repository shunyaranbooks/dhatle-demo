import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dhatle.constants import STAGES, STAGE_INDEX, DEFAULT_GATES
from dhatle.engine import TalentLifecycleEngine, EmployeeState
from dhatle.agents import AgenticLayer, Governance
from dhatle.fairness import group_fairness_score, resignation_risk
from dhatle.explain import train_promotion_model, shap_explain
from dhatle.governance import maturity_score, autopilot_enabled
from dhatle.utils import append_audit, new_version_hash

st.set_page_config(page_title="DHATLE Demo", layout="wide")

st.title("DHATLE — Dynamic Human–Agentic Talent Lifecycle Ecosystem (Demo)")

@st.cache_data
def load_data():
    path = os.path.join("dhatle","data","synthetic_employees.csv")
    df = pd.read_csv(path)
    return df

df = load_data()

# Session init
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

engine = st.session_state.engine
agents = st.session_state.agents
gov = st.session_state.gov
model = st.session_state.model

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

tabs = st.tabs(["Overview", "Employee Decisions", "Fairness & Risk", "Explainability", "Governance & Audit"])

with tabs[0]:
    st.subheader("Workforce overview")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="current_stage", title="Employees per Stage")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="dept", color="gender", barmode="group", title="Dept by Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head(20))

with tabs[1]:
    st.subheader("Human–Agent Decisions")
    emp_ids = df["emp_id"].tolist()
    emp_id = st.selectbox("Select employee", emp_ids, index=0)
    row = df[df.emp_id==emp_id].iloc[0]
    e = EmployeeState(emp_id=emp_id, stage=row.current_stage,
                      performance=row.performance, engagement=row.engagement,
                      learning_speed=row.learning_speed)

    phi_row = agents.get_row(emp_id, e.stage)
    next_stage, probs = engine.decide_next(e, phi_row)

    st.markdown(f"**Current stage:** `{e.stage}` → **Proposed:** `{next_stage}`")
    st.json({k: round(v,4) for k,v in probs.items()})

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Performance", f"{e.performance:.2f}")
    with colB:
        st.metric("Engagement", f"{e.engagement:.2f}")
    with colC:
        st.metric("Learning Speed", f"{e.learning_speed:.2f}")

    st.markdown("### Manager feedback")
    approved = st.radio("Approve AI decision?", ["Approve","Override"], horizontal=True)
    note = st.text_input("Notes / reason (optional)")
    if approved=="Override":
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
        agents.feedback_update(emp_id, final_stage, fb)
        gov.record_feedback(emp_id, approved_bool)
        # append audit
        version = new_version_hash({"emp_id": emp_id, "from": e.stage, "to": final_stage, "probs": probs})
        append_audit({
            "version": version,
            "actor": "manager",
            "emp_id": emp_id,
            "action": "approve" if approved_bool else "override",
            "from_stage": e.stage,
            "to_stage": final_stage,
            "prob": round(float(probs.get(final_stage, 0.0)),4),
            "approved": approved_bool,
            "feedback": fb,
            "notes": note or ""
        })
        # update in df for continuity
        df.loc[df.emp_id==emp_id, "current_stage"] = final_stage
        st.success("Decision recorded and learning updated.")

with tabs[2]:
    st.subheader("Group Fairness & Resignation Risk")
    # Build promotions from model as proxy for opportunity
    X = df[["gender","dept","age","tenure_months","performance","engagement","learning_speed"]]
    y_pred = (st.session_state.model.predict_proba(X)[:,1] > 0.5).astype(int)
    y_true = y_pred  # proxy (demo)

    fairness = group_fairness_score(df, y_true, y_pred, sensitive_col="gender")
    st.json(fairness)

    # Compute RR per employee using group fairness for their gender
    group_to_F = {}
    # Convert fairness to per-group by reusing rates: higher disparity => lower F
    max_rate = max(fairness["rates"].values()) if fairness["rates"] else 1.0
    for g, rate in fairness["rates"].items():
        group_to_F[g] = rate / max_rate if max_rate>0 else 1.0

    gammas = (g1,g2,g3)
    df["RR"] = [
        resignation_risk(row.engagement, row.performance, group_to_F.get(str(row.gender), fairness["fairness"]), gammas)
        for _, row in df.iterrows()
    ]
    st.write("Top 10 resignation risks")
    st.dataframe(df.sort_values("RR", ascending=False).head(10)[["emp_id","gender","dept","performance","engagement","RR"]])

    fig = px.histogram(df, x="RR", nbins=20, title="Distribution of Resignation Risk")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Explainability")
    st.caption("SHAP for promotion propensity model")
    idxs = st.multiselect("Pick up to 5 employees to explain", list(range(len(df))), max_selections=5, default=[0,1,2])
    if idxs:
        with st.spinner("Computing SHAP…"):
            shap_vals, base = shap_explain(st.session_state.model, df, idxs)
        st.write("Expected value (base):", base)
        st.write("SHAP shape:", np.array(shap_vals).shape)
        st.info("Use local run `pip install shap` with jupyter to render full SHAP plots, or integrate with plotly for bars. For demo, we show the array shape to confirm computation.")

with tabs[4]:
    st.subheader("Governance & Audit")
    emp_id = st.selectbox("Employee for maturity check", df.emp_id.unique(), index=0, key="gov_emp")
    score = maturity_score(gov, emp_id)
    st.metric("Maturity Score", f"{score:.2f}")
    st.write("Autopilot:", "Enabled" if autopilot_enabled(score, threshold=mth) else "Disabled")
    st.download_button("Download audit log CSV", data=open("audit_log.csv","rb").read() if os.path.exists("audit_log.csv") else b"", file_name="audit_log.csv", disabled=not os.path.exists("audit_log.csv"))
    st.caption("Audit log is append-only with version hashes for each decision.")
