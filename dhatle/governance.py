from typing import Dict
from .agents import Governance

def maturity_score(gov: Governance, emp_id: str, omega=(0.5,0.3,0.2)) -> float:
    return gov.maturity(emp_id, w=omega)

def autopilot_enabled(score: float, threshold: float=0.75) -> bool:
    return score >= threshold
