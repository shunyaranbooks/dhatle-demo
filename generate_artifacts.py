"""
generate_artifacts.py — DHATLE report generator (CSV+PNG only)
--------------------------------------------------------------
Place this file in your repo at: /dhatle-demo/generate_artifacts.py
Then run:
    python generate_artifacts.py

Outputs to /dhatle-demo/output/:
- Figures (PNG): fig1_architecture.png, fig2_tle_state_machine.png,
                 fig3_rhml_phi.png, fig4_fairness_rates.png,
                 fig5_rr_hist.png, fig6_shap_waterfall.png, fig6_shap_mean_abs.png
- Tables (CSV):  table1_notation.csv, table2_hparams.csv,
                 table3_dataset.csv, table4_traditional_vs_dhatle.csv,
                 table5_sensitivity.csv
- README.md and manifest.json

All tables are **CSV** (no LaTeX). All figures are **PNG**.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from typing import List, Dict, Any

# Reuse your existing codebase (must run from /dhatle-demo)
from dhatle.constants import STAGES, STAGE_INDEX, DEFAULT_GATES
from dhatle.engine import TalentLifecycleEngine, EmployeeState
from dhatle.agents import AgenticLayer, Governance
from dhatle.fairness import group_fairness_score, resignation_risk
from dhatle.explain import train_promotion_model, shap_explain
from dhatle.governance import maturity_score, autopilot_enabled

# -----------------------------
# Paths
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "dhatle", "data", "synthetic_employees.csv")
OUT_DIR = os.path.join(ROOT, "output")

# -----------------------------
# Utilities
# -----------------------------
def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def save_csv(filename: str, df: pd.DataFrame) -> str:
    ensure_out()
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[OK] Wrote {path}")
    return path

def save_md(filename: str, content: str) -> str:
    ensure_out()
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"[OK] Wrote {path}")
    return path

def save_fig(filename: str, fig: plt.Figure) -> str:
    ensure_out()
    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {path}")
    return path

# -----------------------------
# Data & Model
# -----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    return df

def train_model(df: pd.DataFrame):
    model, acc = train_promotion_model(df)
    return model, acc

# -----------------------------
# Figures
# -----------------------------
def fig_architecture() -> str:
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")

    def box(x, y, w, h, label):
        rect = Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center")
        return rect

    def arrow(p1, p2, dashed=False):
        style = "--" if dashed else "-"
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=2, linestyle=style)
        ax.add_patch(a)

    # Boxes
    box(3.0, 3.0, 3.0, 1.2, "Talent Lifecycle Engine\n(State Machine + Gates)")
    box(0.5, 5.0, 3.0, 1.2, "Agentic AI Layer\n(φ proposals, tasks)")
    box(6.5, 5.0, 3.0, 1.2, "Reciprocal Human–AI\nLearning (RHML)")
    box(0.5, 1.0, 3.0, 1.2, "Ethical & Fairness\nOversight (metrics, RR)")
    box(3.0, 0.2, 3.0, 1.2, "Explainability & Trust\n(SHAP, reasons)")
    box(6.5, 1.0, 3.0, 1.2, "Governance & Adaptation\n(maturity, policy, audit)")

    # Arrows
    arrow((2.0, 5.6), (3.0, 3.6))           # Agent -> Core
    arrow((8.0, 5.6), (6.0, 3.6))           # RHML -> Core
    arrow((2.0, 1.6), (3.0, 3.0))           # Fair -> Core
    arrow((4.5, 2.0), (4.5, 3.0))           # Core -> Explainability
    arrow((8.0, 1.6), (6.0, 3.0))           # Governance -> Core
    arrow((6.0, 0.8), (7.2, 5.0), dashed=True)  # Explainability -> RHML

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    return save_fig("fig1_architecture.png", fig)

def fig_tle_state_machine() -> str:
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    ax.axis("off")

    # Horizontal boxes for six stages
    x0, y, w, h, gap = 0.2, 0.6, 1.5, 0.9, 0.25
    boxes = []
    for name in STAGES:
        xi = x0 + len(boxes) * (w + gap)
        rect = Rectangle((xi, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        txt = name.replace("_", " ").replace("And", "&")
        ax.text(xi + w/2, y + h/2, txt, ha="center", va="center", fontsize=9)
        boxes.append((xi, y, w, h))

    def arrow(x1, x2):
        a = FancyArrowPatch((x1, y + h/2), (x2, y + h/2),
                            arrowstyle="-|>", mutation_scale=12, linewidth=2)
        ax.add_patch(a)

    # Forward arrows
    for i in range(len(boxes) - 1):
        arrow(boxes[i][0] + boxes[i][2], boxes[i+1][0])

    # Remedial loop (ENG -> L&D)
    from matplotlib.patches import FancyArrowPatch
    x_eng = boxes[4][0] + boxes[4][2]
    a = FancyArrowPatch((x_eng - 0.2, y + 0.1), (boxes[2][0] - 0.2, y + 1.6),
                        arrowstyle="-|>", mutation_scale=12, linewidth=2,
                        connectionstyle="arc3,rad=0.3")
    ax.add_patch(a)

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 3)
    return save_fig("fig2_tle_state_machine.png", fig)

def fig_rhml_phi(df: pd.DataFrame) -> str:
    # Pick one employee, show φ shift after approvals
    row = df.iloc[0]
    emp_id = row["emp_id"]
    current_stage = row["current_stage"]
    ag = AgenticLayer()

    # before
    row_before = ag.get_row(emp_id, current_stage).copy()

    # simulate approvals to bias toward next stage
    idx = STAGE_INDEX[current_stage]
    target = STAGES[min(idx + 1, len(STAGES) - 1)]
    for _ in range(5):
        ag.feedback_update(emp_id, target, 1.0)
    row_after = ag.get_row(emp_id, current_stage).copy()

    xs = np.arange(len(STAGES))
    before = np.array([row_before.get(s, 0.0) for s in STAGES])
    after = np.array([row_after.get(s, 0.0) for s in STAGES])

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.35
    ax.bar(xs - width/2, before, width=width, label="Before feedback")
    ax.bar(xs + width/2, after,  width=width, label="After RHML")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_", " ") for s in STAGES], rotation=15, ha="right")
    ax.set_ylabel("Probability mass (φ)")
    ax.set_title(f"RHML effect on φ (employee {emp_id})")
    ax.legend()
    return save_fig("fig3_rhml_phi.png", fig)

def fig_fairness_rates(df: pd.DataFrame, model) -> str:
    feature_cols = ["gender","dept","age","tenure_months","performance","engagement","learning_speed"]
    X = df[feature_cols]
    y_pred = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
    y_true = y_pred  # proxy for demo
    fairness = group_fairness_score(df, y_true, y_pred, sensitive_col="gender")

    groups = list(fairness["rates"].keys())
    rates = [fairness["rates"][g] for g in groups]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(np.arange(len(groups)), rates)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Selection rate")
    ax.set_title(f"Selection rates by gender (DPD ≈ {fairness['dp_diff']:.3f})")
    return save_fig("fig4_fairness_rates.png", fig)

def fig_rr_hist(df: pd.DataFrame, model, gammas=(0.4, 0.4, 0.2)) -> str:
    # Map fairness to F(i)
    feature_cols = ["gender","dept","age","tenure_months","performance","engagement","learning_speed"]
    X = df[feature_cols]
    y_pred = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
    fairness = group_fairness_score(df, y_pred, y_pred, sensitive_col="gender")
    max_rate = max(fairness["rates"].values()) if fairness["rates"] else 1.0
    group_to_F = {g: (r / max_rate if max_rate > 0 else 1.0) for g, r in fairness["rates"].items()}

    rr = []
    for _, r in df.iterrows():
        rr.append(resignation_risk(
            float(r.engagement),
            float(r.performance),
            group_to_F.get(str(r.gender), fairness["fairness"]),
            gammas
        ))
    rr = np.array(rr)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(rr, bins=20)
    ax.set_xlabel("Resignation risk (RR)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of RR (γ={gammas})")
    return save_fig("fig5_rr_hist.png", fig)

def fig_shap(df: pd.DataFrame, model) -> List[str]:
    feature_cols = ["gender","dept","age","tenure_months","performance","engagement","learning_speed"]
    idxs = [0]  # explain first row
    try:
        shap_vals, base = shap_explain(model, df, idxs)
        v = np.array(shap_vals)[0]

        import shap as _shap
        exp = _shap.Explanation(values=v, base_values=float(base),
                                data=df[feature_cols].iloc[idxs[0]].values,
                                feature_names=feature_cols)

        fig1 = plt.figure(figsize=(7.5, 5.0))
        _shap.plots.waterfall(exp, show=False, max_display=12)
        p1 = save_fig("fig6_shap_waterfall.png", fig1)

        mean_abs = np.abs(v)
        order = np.argsort(mean_abs)
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        ax2.barh(np.array(feature_cols)[order], mean_abs[order])
        ax2.set_xlabel("|SHAP| (log-odds)")
        ax2.set_title("Mean |SHAP| across selected employees")
        p2 = save_fig("fig6_shap_mean_abs.png", fig2)
        return [p1, p2]

    except Exception as e:
        # Fallback image with guidance
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.axis("off")
        ax.text(0.02, 0.6, "SHAP rendering unavailable.\nInstall `shap` to generate plots.",
                fontsize=12, va="center")
        ax.text(0.02, 0.3, f"Error: {type(e).__name__}: {e}", fontsize=9, va="center")
        p = save_fig("fig6_shap_waterfall.png", fig)
        # Also create an empty mean-abs placeholder
        fig2, ax2 = plt.subplots(figsize=(7.5, 2.0))
        ax2.axis("off")
        p2 = save_fig("fig6_shap_mean_abs.png", fig2)
        return [p, p2]

# -----------------------------
# Tables (CSV)
# -----------------------------
def table1_notation() -> str:
    rows = [
        ("S_t(i)", "Employee i's lifecycle stage at time t"),
        ("P_t(i), G_t(i), L(i)", "Performance, Engagement, Learning speed"),
        ("θ", "Gate thresholds / policy parameters"),
        ("φ_{jk}(i)", "Probability of s_k → s_j for employee i"),
        ("λ", "RHML learning rate"),
        ("RR_t(i)", "Resignation risk at time t"),
        ("γ_1, γ_2, γ_3", "Weights in RR_t model"),
        ("ω_1, ω_2, ω_3", "Maturity weights (Accuracy / Trust / Stability)"),
    ]
    df = pd.DataFrame(rows, columns=["Symbol", "Meaning"])
    return save_csv("table1_notation.csv", df)

def table2_hparams(lambda_default=0.2, gammas=(0.4, 0.4, 0.2), maturity_thr=0.75) -> str:
    rows = [
        ("perf_min", f"{DEFAULT_GATES.get('perf_min', 0.5):.2f}", "Minimum performance to advance"),
        ("eng_min", f"{DEFAULT_GATES.get('eng_min', 0.5):.2f}", "Minimum engagement to advance"),
        ("learn_min", f"{DEFAULT_GATES.get('learn_min', 0.45):.2f}", "Minimum learning speed to advance"),
        ("λ", f"{lambda_default:.2f}", "RHML learning rate"),
        ("(γ1, γ2, γ3)", f"({gammas[0]:.1f}, {gammas[1]:.1f}, {gammas[2]:.1f})", "RR weights"),
        ("Maturity threshold", f"{maturity_thr:.2f}", "Autopilot enable cutoff"),
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Default", "Description"])
    return save_csv("table2_hparams.csv", df)

def table3_dataset(df: pd.DataFrame) -> str:
    genders = list(map(str, sorted(df["gender"].unique().tolist())))
    depts = list(map(str, sorted(df["dept"].unique().tolist())))
    n = len(df)
    tmin, tmax = int(df["tenure_months"].min()), int(df["tenure_months"].max())
    rows = [
        ("Gender categories", "/".join(genders)),
        ("Departments (count)", f"{len(depts)}"),
        ("Departments (list)", ", ".join(depts)),
        ("Tenure months (min–max)", f"{tmin}–{tmax}"),
        ("Signals scaled", "P, G, L ∈ [0,1]"),
        ("Employees (N)", f"{n}"),
    ]
    out = pd.DataFrame(rows, columns=["Attribute", "Value"])
    return save_csv("table3_dataset.csv", out)

def table4_comparison() -> str:
    rows = [
        ("Lifecycle Coverage", "Single step (e.g., hiring)", "Full journey (hire → succession)"),
        ("Model Structure", "Often black-box", "Transparent state-transition"),
        ("Signals (P,G,L)", "Rarely modeled", "Explicitly modeled"),
        ("Attrition Risk", "Simple binary", "Multi-variate, fairness-aware"),
        ("Agent Learning", "Static", "Reciprocal feedback (RHML)"),
        ("Fairness", "Post-hoc", "Built-in to core loop"),
        ("Explainability", "Limited", "Dashboards, reasons, SHAP"),
        ("Governance", "Weak", "Maturity, policy, audit trail"),
    ]
    df = pd.DataFrame(rows, columns=["Feature", "Traditional Systems", "DHATLE System"])
    return save_csv("table4_traditional_vs_dhatle.csv", df)

def simulate_forward_transitions(df: pd.DataFrame, perf_min, eng_min, learn_min, lam) -> Dict[str, Any]:
    """One sweep step using actual engine/agents/model."""
    eng = TalentLifecycleEngine()
    ag = AgenticLayer()
    ag.learning_rate = lam
    eng.gates.update({"perf_min": perf_min, "eng_min": eng_min, "learn_min": learn_min})

    # Model (aligns with app)
    model, _ = train_promotion_model(df)

    forward_count, total = 0, 0
    for _, r in df.iterrows():
        e = EmployeeState(
            emp_id=r.emp_id,
            stage=r.current_stage,
            performance=float(r.performance),
            engagement=float(r.engagement),
            learning_speed=float(r.learning_speed),
        )
        row_phi = ag.get_row(e.emp_id, e.stage)
        next_stage, probs = eng.decide_next(e, row_phi)
        if STAGE_INDEX[next_stage] > STAGE_INDEX[e.stage]:
            forward_count += 1
        total += 1
        fb = 1.0 if STAGE_INDEX[next_stage] > STAGE_INDEX[e.stage] else 0.0
        ag.feedback_update(e.emp_id, next_stage, fb)

    fwd_rate = forward_count / max(1, total)

    # Fairness & RR
    feature_cols = ["gender","dept","age","tenure_months","performance","engagement","learning_speed"]
    X = df[feature_cols]
    y_pred = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
    fairness = group_fairness_score(df, y_pred, y_pred, sensitive_col="gender")
    max_rate = max(fairness["rates"].values()) if fairness["rates"] else 1.0
    group_to_F = {g: (r / max_rate if max_rate > 0 else 1.0) for g, r in fairness["rates"].items()}

    gammas = (0.4, 0.4, 0.2)
    rr = []
    for _, r in df.iterrows():
        rr.append(resignation_risk(
            float(r.engagement), float(r.performance),
            group_to_F.get(str(r.gender), fairness["fairness"]),
            gammas
        ))
    rr = np.array(rr)
    mean_rr = float(rr.mean())
    top10_rr = float(np.sort(rr)[-10:].mean()) if len(rr) >= 10 else float(rr.mean())

    # Proxy maturity (same structure as paper; heuristic stability)
    trust = fwd_rate if total > 0 else 0.5
    acc_prior = 0.75
    accuracy = 0.5 * acc_prior + 0.5 * trust
    stability = 1.0 / (1.0 + 0.5 * (1.0 - trust))
    maturity = 0.5 * accuracy + 0.3 * trust + 0.2 * stability

    return {
        "forward_rate": fwd_rate,
        "dp_diff": float(fairness["dp_diff"]),
        "mean_rr": mean_rr,
        "top10_rr": top10_rr,
        "maturity": maturity,
    }

def table5_sensitivity(df: pd.DataFrame) -> str:
    settings = [
        ("perf_min=0.50", {"perf_min": 0.50, "eng_min": DEFAULT_GATES["eng_min"], "learn_min": DEFAULT_GATES["learn_min"], "lam": 0.20}),
        ("perf_min=0.70", {"perf_min": 0.70, "eng_min": DEFAULT_GATES["eng_min"], "learn_min": DEFAULT_GATES["learn_min"], "lam": 0.20}),
        ("eng_min=0.60",  {"perf_min": DEFAULT_GATES["perf_min"], "eng_min": 0.60, "learn_min": DEFAULT_GATES["learn_min"], "lam": 0.20}),
        ("λ=0.10",        {"perf_min": DEFAULT_GATES["perf_min"], "eng_min": DEFAULT_GATES["eng_min"], "learn_min": DEFAULT_GATES["learn_min"], "lam": 0.10}),
        ("λ=0.30",        {"perf_min": DEFAULT_GATES["perf_min"], "eng_min": DEFAULT_GATES["eng_min"], "learn_min": DEFAULT_GATES["learn_min"], "lam": 0.30}),
    ]
    rows = []
    for name, p in settings:
        r = simulate_forward_transitions(df, p["perf_min"], p["eng_min"], p["learn_min"], p["lam"])
        rows.append({
            "Setting": name,
            "Fwd Transitions": round(r["forward_rate"], 3),
            "DP Diff": round(r["dp_diff"], 3),
            "Mean RR": round(r["mean_rr"], 3),
            "Top10 RR": round(r["top10_rr"], 3),
            "Maturity": round(r["maturity"], 3),
        })
    df_out = pd.DataFrame(rows, columns=["Setting","Fwd Transitions","DP Diff","Mean RR","Top10 RR","Maturity"])
    return save_csv("table5_sensitivity.csv", df_out)

# -----------------------------
# README & Manifest
# -----------------------------
def write_readme(model_acc: float, outputs: Dict[str, str]):
    md = f"""# DHATLE Generated Artifacts
Date: {datetime.datetime.utcnow().isoformat()} UTC

Model test accuracy: **{model_acc:.3f}**

## Figures (PNG)
- {os.path.basename(outputs['fig1'])}
- {os.path.basename(outputs['fig2'])}
- {os.path.basename(outputs['fig3'])}
- {os.path.basename(outputs['fig4'])}
- {os.path.basename(outputs['fig5'])}
- {os.path.basename(outputs['fig6a'])}
- {os.path.basename(outputs['fig6b'])}

## Tables (CSV)
- table1_notation.csv
- table2_hparams.csv
- table3_dataset.csv
- table4_traditional_vs_dhatle.csv
- table5_sensitivity.csv
"""
    return save_md("README.md", md)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_out()
    df = load_data()
    model, acc = train_model(df)

    # Figures
    f1 = fig_architecture()
    f2 = fig_tle_state_machine()
    f3 = fig_rhml_phi(df)
    f4 = fig_fairness_rates(df, model)
    f5 = fig_rr_hist(df, model)
    f6a, f6b = fig_shap(df, model)

    # Tables (CSV)
    t1 = table1_notation()
    t2 = table2_hparams()
    t3 = table3_dataset(df)
    t4 = table4_comparison()
    t5 = table5_sensitivity(df)

    # README + manifest
    outputs = {
        "fig1": f1, "fig2": f2, "fig3": f3, "fig4": f4, "fig5": f5, "fig6a": f6a, "fig6b": f6b,
        "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5
    }
    write_readme(acc, outputs)

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"[OK] Wrote {manifest_path}")
    print("\nAll artifacts generated in:", OUT_DIR)

if __name__ == "__main__":
    main()
