from typing import Dict
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

def group_fairness_score(df: pd.DataFrame, y_true, y_pred, sensitive_col: str) -> Dict[str, float]:
    mf = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred,
                     sensitive_features=df[sensitive_col])
    # Score closer to 1 is better (lower disparity)
    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=df[sensitive_col])
    # Clamp and invert: smaller difference -> higher fairness
    fairness_overall = 1.0 - float(min(1.0, max(0.0, abs(dp_diff))))
    # per-group rates
    rates = {str(k): float(v) for k,v in mf.by_group.items()}
    return {"fairness": fairness_overall, "rates": rates, "dp_diff": float(dp_diff)}

def resignation_risk(G_t: float, P_t: float, F_group: float, gammas=(0.4,0.4,0.2), eta_std=0.03) -> float:
    # RR_t(i) = γ1(1 − G_t(i)) + γ2(1 − P_t(i)) + γ3 × F(i) + η_t
    eta = np.random.normal(0, eta_std)
    rr = gammas[0]*(1.0-G_t) + gammas[1]*(1.0-P_t) + gammas[2]*(1.0 - F_group) + eta
    return float(np.clip(rr, 0.0, 1.0))
