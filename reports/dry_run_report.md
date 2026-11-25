# Dry-run Fase 5 - catena riproducibile

Data: 2026-11-25 17:59:05

## Esecuzione script

| script | esito | ultimo log |
|---|---|---|
| fase2_acquisizione_preprocessing.py | PASS | Fase 2 completata (task automatizzabili). Vedi CHECKLIST.md per lo stato dei task manuali. |
| fase3_analisi_statistica.py | PASS | Fase 3 completata (task 3.5/3.6 bloccati per assenza dati geochimici). |
| fase4_reporting.py | PASS | Fase 4 completata (4.2 PDF via Docker resta pending). |

## Validazione artefatti

| check | esito |
|---|---|
| sha256 gossip_20260217_20260218.csv | PASS |
| sha256 ingv_events.csv | PASS |
| sha256 seismic_ingv_20250525_20260525.csv | PASS |
| sha256 uplift.csv | PASS |
| artefatto data/processed/catalog_gossip_20260217_20260218.csv | PASS |
| artefatto data/processed/catalog_clean_ingv_events.csv | PASS |
| artefatto data/processed/uplift_clean.csv | PASS |
| artefatto data/processed/gnss_changepoints.csv | PASS |
| artefatto data/processed/csd_gnss_rolling.csv | PASS |
| artefatto data/processed/iet_weibull.csv | PASS |
| artefatto data/processed/hawkes_etas_fit.csv | PASS |
| artefatto data/processed/montecarlo_stress_threshold.csv | PASS |
| artefatto reports/Flegrei_Alert_2026.md | PASS |
| artefatto reports/units_check.csv | PASS |
| artefatto reports/fig/fig1_uplift_changepoints.svg | PASS |
| artefatto reports/fig/fig2_csd_rolling.svg | PASS |
| artefatto reports/fig/fig3_weibull_hazard.svg | PASS |
| artefatto reports/fig/fig4_phase_space.svg | PASS |
| DataPackage.zip integrita' (0.64 MB, limite 10 MB) | PASS |

**Esito complessivo: PASS**

Nota: la parte 'Docker start -> PDF' del task 5.1 richiede daemon Docker + immagine R 4.4 (non disponibile in CI sandbox); resta pending e va eseguita via MANUAL_EXECUTION.md §§1-4.
