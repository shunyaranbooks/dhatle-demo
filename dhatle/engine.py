from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .constants import STAGES, STAGE_INDEX, DEFAULT_GATES

@dataclass
class EmployeeState:
    emp_id: str
    stage: str
    performance: float
    engagement: float
    learning_speed: float

@dataclass
class TalentLifecycleEngine:
    gates: Dict[str, float] = field(default_factory=lambda: DEFAULT_GATES.copy())
    history: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def gate_pass(self, e: EmployeeState) -> Dict[str, bool]:
        return {
            "perf": e.performance >= self.gates["perf_min"],
            "eng": e.engagement >= self.gates["eng_min"],
            "learn": e.learning_speed >= self.gates["learn_min"],
        }

    def allowed_transitions(self, current_stage: str) -> List[str]:
        idx = STAGE_INDEX[current_stage]
        # forward move or stay; allow skip to Engagement for low eng
        options = [current_stage]
        if idx < len(STAGES)-1:
            options.append(STAGES[idx+1])
        if current_stage == "Performance_Management":
            options.append("Engagement")
        if current_stage == "Engagement":
            options.append("Learning_Development")
        return list(dict.fromkeys(options))

    def decide_next(self, e: EmployeeState, phi_row: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        allowed = self.allowed_transitions(e.stage)
        # re-normalize probabilities over allowed
        probs = {s: max(1e-6, phi_row.get(s, 0.0)) for s in allowed}
        s = sum(probs.values())
        probs = {k: v/s for k,v in probs.items()}
        # apply gates: if a gate fails, prevent forward move
        gp = self.gate_pass(e)
        if not all(gp.values()) and e.stage in allowed:
            # bias to stay if gates not all passed
            stay = e.stage
            probs = {k: (0.1 if k!=stay else 0.9) for k in probs}
        # choose argmax
        next_stage = max(probs, key=probs.get)
        self.history.setdefault(e.emp_id, []).append((e.stage, next_stage))
        return next_stage, probs
