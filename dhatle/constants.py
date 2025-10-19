from enum import IntEnum

STAGES = [
    "Attraction_Acquisition",
    "Onboarding",
    "Learning_Development",
    "Performance_Management",
    "Engagement",
    "Retention_Succession"
]

STAGE_INDEX = {name: i for i, name in enumerate(STAGES)}

class Stage(IntEnum):
    Attraction_Acquisition = 0
    Onboarding = 1
    Learning_Development = 2
    Performance_Management = 3
    Engagement = 4
    Retention_Succession = 5

DEFAULT_GATES = {
    # min thresholds to move forward
    "perf_min": 0.5,
    "eng_min": 0.5,
    "learn_min": 0.45,
    # noise level for RR eta
    "rr_noise": 0.03
}
