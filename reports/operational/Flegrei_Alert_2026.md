# Flegrei Alert 2026 - Nota Tecnica

## Abstract

This technical note presents an integrated quantitative monitoring analysis of the Campi Flegrei caldera, combining GNSS ground-deformation (uplift) and seismic-catalog data acquired from public INGV services. Data acquisition and preprocessing (Phase 2) produced a validated raw archive with SHA-256 checksums, a GOSSIP relocated-catalog extract for 17-18 February 2026 (events #51673, #51676, #51678, #51684), and cleaned catalogs. Advanced statistical analysis (Phase 3) detected multiple change-points in the GNSS uplift series (PELT, cross-checked with Bai-Perron binary segmentation, bootstrap 95% CI with B=1000), quantified the CSD rolling variance and lag-1 autocorrelation, fitted a Weibull distribution to inter-event times (shape beta = 0.367 < 1, indicating clustered seismicity), and confirmed self-excitation with a Hawkes/ETAS model (AIC 24.5 vs 360.4 for a homogeneous Poisson process on the same 180-day window). Monte-Carlo uncertainty propagation (B=1000) on the stress threshold (yield point = 0.869 m, 95% CI [0.847, 0.893]) estimates P(margin>0) = 0.993 and P(exceeding yield + 2 sigma within 30 days) = 0.264. Geochemical analyses (cointegration and Granger causality) remain pending due to the lack of machine-readable INGV-OV gas bulletins.

## 1. Dati (Fase 2)

- Serie uplift GNSS: 365 giorni (data/raw/uplift.csv, SHA-256 in data/raw/SHA256SUMS.txt)
- Catalogo GOSSIP 17-18/02/2026: 4 eventi (data/raw/gossip_20260217_20260218.csv)
- Catalogo INGV pulito: 2505 eventi 2008-2026 in bounding box Campi Flegrei (data/processed/catalog_clean_ingv_events.csv)
- Outlier IQR su uplift: 0/365 (data/processed/uplift_outliers.csv)

## 2. Analisi (Fase 3)

### 2.1 Geodesia: change-point

| cp | data | media prima | media dopo | CI95% (bootstrap B=1000) |
|---|---|---|---|---|
| 53 | 2025-07-14 | 0.5147 | 0.5770 | 2025-06-21 - 2025-08-06 |
| 105 | 2025-09-04 | 0.5770 | 0.6309 | 2025-08-19 - 2025-09-25 |
| 156 | 2025-10-25 | 0.6309 | 0.6884 | 2025-10-10 - 2025-11-17 |
| 209 | 2025-12-17 | 0.6884 | 0.9016 | 2025-12-04 - 2026-01-04 |
| 210 | 2025-12-18 | 0.9016 | 0.7350 | 2026-01-17 - 2026-01-17 |
| 266 | 2026-02-12 | 0.7350 | 0.7813 | 2026-01-26 - 2026-03-14 |
| 296 | 2026-03-14 | 0.7813 | 0.8264 | 2026-03-15 - 2026-04-10 |
| 350 | 2026-05-07 | 0.8264 | 0.8693 | 2026-04-17 - 2026-05-18 |

Figura: [fig1_uplift_changepoints.svg](fig/fig1_uplift_changepoints.svg), [fig2_csd_rolling.svg](fig/fig2_csd_rolling.svg)

### 2.2 Sismologia: Weibull sugli IET e Hawkes/ETAS

- Weibull IET: beta = 0.3671, lambda = 7.0104 h, AIC = 15150.61 (vs esponenziale 25752.80); beta < 1 -> hazard decrescente (clustering)
- Hawkes/ETAS (180 gg, 191 eventi): mu=0.529520/giorno, K=0.022472, alpha=0.3980/Mw, c=0.000520 gg, p=1.0904; AIC Hawkes=24.52 vs Poisson=360.45 -> auto-eccitazione confermata

Figura: [fig3_weibull_hazard.svg](fig/fig3_weibull_hazard.svg), [fig4_phase_space.svg](fig/fig4_phase_space.svg)

### 2.3 Integrazione: Monte-Carlo sulle soglie di stress

| metrica | valore | CI95% |
|---|---|---|
| yield_level | 0.869286 | 0.847127 - 0.892743 |
| margin_last_vs_yield | 0.029221 | 0.006509 - 0.052019 |
| p_margin_positive | 0.993000 |  |
| p_exceed_yield_plus_2sigma_in_30d | 0.264000 |  |
| post_cp_drift_per_day | 0.001082 |  |
| sigma_pre_cp | 0.113822 |  |

## 3. Unita' di misura

Verifica completa in [units_check.csv](units_check.csv). Nota: l'unita' di `uplift` non e' dichiarata nella sorgente (assunto m cumulati; da verificare vs INGV).

## 4. Limiti

- Task 2.4/3.5/3.6 (geochimica CO2/H2O, cointegrazione, Granger) bloccati: bollettini INGV-OV senza endpoint machine-readable (voir data/processed/geochemistry_status.csv)
- Catalogo GOSSIP: evento #51675 non presente nella finestra raw originale
- Confronto AIC Hawkes vs Weibull su likelihood di rappresentazioni diverse: il verdetto di auto-eccitazione si basa sul confronto Hawkes vs Poisson (stessa likelihood)

## 5. Riproducibilita'

```bash
python3 legacy-flegrei-alert-2026/scripts/fase2_acquisizione_preprocessing.py
python3 legacy-flegrei-alert-2026/scripts/fase3_analisi_statistica.py
python3 legacy-flegrei-alert-2026/scripts/fase4_reporting.py
# knit PDF (richiede Docker R 4.4 + LaTeX, vedi MANUAL_EXECUTION.md):
docker run --rm -v "$PWD:/app" flegrei-alert Rscript -e "rmarkdown::render('legacy-flegrei-alert-2026/notebook/report.Rmd')"
```
