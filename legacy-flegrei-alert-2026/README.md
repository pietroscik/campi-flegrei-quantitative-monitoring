# Legacy: Flegrei_Alert_2026

This folder consolidates the unique content of `Flegrei_Alert_2026` (now a private repository), which documented the same Campi Flegrei monitoring system with an operational (IT-language) focus while this repository carries the scientific/production version (CI, tests, CITATION).

## Contents

- `scripts/` — standalone operational scripts (b-value, anomaly detection, ETAS/Hawkes, GNSS and geochemistry in R, early warning, multi-signal fusion); complementary to the modules in `src/`.
- `notebook/report.Rmd` — RMarkdown analysis report.
- `CHECKLIST.md`, `MANUAL_EXECUTION.md`, `metadata.yaml` — operational runbook and experiment metadata.
- `run_pipeline.py`, `Dockerfile`, `src/analysis/*.py` — the Flegrei_Alert_2026 versions of files that also exist at the repo root; in some cases they were more advanced (e.g. the Docker image with R 4.4 + tidyverse). They are preserved here for reference and are **not wired into the current pipeline**: the root versions remain the source of truth.
- `doc_full/` — technical documents (reproducible-workflow guide in VSCode, investigation report).
- `vscode-tasks.json` — editor task configuration from the original workspace.

## Related data

- `data/raw/seismic_ingv_20250525_20260525.csv` — raw INGV seismic catalog (1-year window) used by the legacy pipeline.
