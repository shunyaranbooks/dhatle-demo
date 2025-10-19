from dataclasses import dataclass, field
from typing import Dict, DefaultDict
from collections import defaultdict
import numpy as np

from .constants import STAGES

@dataclass
class AgenticLayer:
    # phi[emp_id][to_stage] = probability
    phi: DefaultDict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    learning_rate: float = 0.2

    def ensure_row(self, emp_id: str, current_stage: str):
        if emp_id not in self.phi or not self.phi[emp_id]:
            # initialize with small bias to move forward else stay
            dist = {s: 0.02 for s in STAGES}
            dist[current_stage] = 0.5
            # small forward bias if exists
            idx = STAGES.index(current_stage)
            if idx < len(STAGES)-1:
                dist[STAGES[idx+1]] = 0.46
            # normalize
            s = sum(dist.values())
            self.phi[emp_id] = {k: v/s for k,v in dist.items()}

    def get_row(self, emp_id: str, current_stage: str) -> Dict[str, float]:
        self.ensure_row(emp_id, current_stage)
        return self.phi[emp_id]

    def feedback_update(self, emp_id: str, to_stage: str, fb_score: float):
        """Reciprocal Human–AI Learning Loop (RHML)
        φ_{jk}(i)(t+1) = φ_{jk}(i)(t) + λ × (FB_t(i) − φ_{jk}(i)(t))
        fb_score in [0,1], applied to proposed to_stage.
        """
        row = self.get_row(emp_id, to_stage)
        old = row.get(to_stage, 0.0)
        new_val = old + self.learning_rate * (fb_score - old)
        # renormalize keeping other probs
        others = {k:v for k,v in row.items() if k!=to_stage}
        scale = max(1e-9, 1.0 - new_val)
        s = sum(others.values())
        if s == 0:
            others = {k: (scale/len(others)) for k in others}
        else:
            others = {k: v * (scale/s) for k,v in others.items()}
        self.phi[emp_id] = {to_stage: new_val, **others}

@dataclass
class Governance:
    approvals: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    overrides: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    # for stability: track total absolute updates to phi per employee
    phi_delta_sum: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record_feedback(self, emp_id: str, approved: bool):
        if approved:
            self.approvals[emp_id] += 1
        else:
            self.overrides[emp_id] += 1

    def track_phi_change(self, emp_id: str, delta: float):
        self.phi_delta_sum[emp_id] += abs(delta)

    def maturity(self, emp_id: str, acc: float = 0.75, w=(0.5,0.3,0.2)) -> float:
        """MaturityScore(i) = ω1*Accuracy + ω2*Trust + ω3*Stability
        Accuracy: proxy via approvals/(approvals+overrides) blended with acc prior
        Trust: approvals rate
        Stability: inverse of normalized phi_delta_sum (bounded)
        """
        a = self.approvals[emp_id]; o = self.overrides[emp_id]
        total = a + o
        trust = (a / total) if total>0 else 0.5
        accuracy = 0.5*acc + 0.5*trust
        # stability: smaller delta -> higher stability
        d = self.phi_delta_sum[emp_id]
        stability = 1.0 / (1.0 + d)
        return w[0]*accuracy + w[1]*trust + w[2]*stability
