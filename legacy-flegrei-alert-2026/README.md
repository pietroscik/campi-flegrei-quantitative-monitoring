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

- `data/raw/seismic_ingv_20250525_20260525.csv` — raw INGV seismic catalog (1-year window, QuakeML) used by the legacy pipeline.
- `data/raw/ingv_events.csv` — raw INGV event CSV catalog (2008 → 2026-05).
- `data/raw/gossip_20260217_20260218.csv` — GOSSIP catalog extract 17–18 Feb 2026 (Fase 2.3: eventi #51673, #51676, #51678, #51684).
- `data/raw/uplift.csv` — INGV GNSS uplift series (Fase 2.1, da `data/external/uplift.csv`).
- `data/raw/SHA256SUMS.txt` — SHA-256 dei file in `data/raw/` (Fase 2.2).
- Fase 3 (analisi statistica, `scripts/fase3_analisi_statistica.py`): output in `data/processed/` — `gnss_changepoints.csv` (PELT + Bai-Perron + bootstrap CI), `csd_gnss_rolling.csv`, `iet_weibull.csv`, `hawkes_etas_fit.csv`, `montecarlo_stress_threshold.csv`, `geochemistry_status.csv` (3.5/3.6 bloccati: no dati gas).
