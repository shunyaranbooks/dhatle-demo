# DHATLE Demo — Dynamic Human–Agentic Talent Lifecycle Ecosystem

This is a runnable, self-contained demo of **DHATLE** aligned with your paper's Section 3 (Architecture → Algorithms). It includes:

- **Talent Lifecycle Engine (TLE)** with transition gates & state memory
- **Agentic AI Layer** with per-employee transition probabilities (φ) and **Reciprocal Human–AI Learning Loop** (RHML)
- **Ethical & Fairness Oversight** (demographic parity, resignation risk RR_t)
- **Explainability & Trust Interface** (promotion propensity model + SHAP)
- **Governance & Adaptation** (maturity score, append-only audit log)

## Run locally

```bash
cd dhatle-demo
pip install -r requirements.txt
streamlit run app.py
```

## Run in GitHub Codespaces

1. Create a Codespace on your repo and open `/dhatle-demo` as workspace folder.
2. VS Code will detect Python; run in terminal:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```
3. Forward port **7860** and open in browser.

## Deploy on Hugging Face Spaces (Streamlit)

- Create a new **Space** → **Streamlit**.
- Point to your GitHub repo and set **`/dhatle-demo`** as the project path.
- It will auto-run `app.py` using `requirements.txt`.

## Files

```
dhatle-demo/
  app.py
  requirements.txt
  README.md
  Dockerfile
  .streamlit/config.toml
  .devcontainer/devcontainer.json
  dhatle/
    __init__.py
    constants.py
    engine.py
    agents.py
    fairness.py
    explain.py
    governance.py
    utils.py
    data/synthetic_employees.csv
  tests/
    test_engine.py
```

## Notes

- **Fairness metrics** use `fairlearn` for demographic parity difference.
- **SHAP** explanations compute values (array). Plotting is minimized in Streamlit for speed; you can expand with `shap.plots.bar` locally.
- **Audit log** persists to `audit_log.csv` in app root (append-only with version hashes).
- **Maturity score** proxies: approvals vs overrides (Trust), a prior Accuracy, and Stability measured by φ updates.

MIT License.
