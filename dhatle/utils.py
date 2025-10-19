import time, hashlib, json, os, csv
from typing import Dict, Any
from datetime import datetime

AUDIT_LOG_PATH = os.environ.get("DHATLE_AUDIT_LOG", "audit_log.csv")

def append_audit(event: Dict[str, Any]):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH) or ".", exist_ok=True)
    exists = os.path.exists(AUDIT_LOG_PATH)
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ts","version","actor","emp_id","action","from_stage","to_stage",
            "prob","approved","feedback","notes"
        ])
        if not exists:
            writer.writeheader()
        event = {"ts": datetime.utcnow().isoformat(), **event}
        writer.writerow(event)

def new_version_hash(payload: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return m.hexdigest()[:12]
