from dhatle.engine import TalentLifecycleEngine, EmployeeState
from dhatle.agents import AgenticLayer

def test_decide_next():
    eng = TalentLifecycleEngine()
    ag = AgenticLayer()
    e = EmployeeState(emp_id="E0001", stage="Onboarding", performance=0.7, engagement=0.7, learning_speed=0.6)
    row = ag.get_row(e.emp_id, e.stage)
    nxt, probs = eng.decide_next(e, row)
    assert nxt in probs and len(probs)>=1
