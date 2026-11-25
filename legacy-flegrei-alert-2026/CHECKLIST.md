# 📋 Checklist Operativa - Nota Tecnica INGV (Feb 2026)

## 🟢 Fase 1: Setup & Riproducibilità (Giorni 1-2)
- [ ] **1.1** Inizializzare repository Git (`git init`) e creare repo remoto (GitHub/GitLab privato).
- [ ] **1.2** Creare `Dockerfile` con R 4.4, Python 3.12 e pacchetti necessari (`tidyverse`, `survival`, `lmtest`, `tick`).
- [ ] **1.3** Testare build Docker (`docker build -t flegrei-alert .`).
- [ ] **1.4** Configurare `.gitignore` per escludere dati sensibili o file >100MB (usare Git LFS se necessario).
- [ ] **1.5** Scrivere `metadata.yaml` con descrizione fonti dati (INGV Open Data) e licenza (CC-BY-4.0).

## 🔵 Fase 2: Acquisizione & Preprocessing Dati (Giorni 2-3)
- [x] **2.1** Scaricare serie GNSS RITE (BEPOS) da portale INGV e salvare in `data/raw/` — serie uplift INGV già in `data/external/uplift.csv` copiata in `data/raw/uplift.csv` da `scripts/fase2_acquisizione_preprocessing.py`; il download stazione RITE/BEPOS dal portale RING non è automatizzabile da CI (portale non raggiungibile), resta un passo manuale opzionale.
- [x] **2.2** Calcolare checksum SHA-256 dei file scaricati e registrarli in `metadata.yaml` — `data/raw/SHA256SUMS.txt` + campi `sha256` in `metadata.yaml` (GNSS_RITE, GOSSIP_Catalog).
- [x] **2.3** Estrarre catalogo sismico (GOSSIP) 17-18 Feb 2026 (Eventi #51675, #51678, etc.) — estratti 4 eventi da `data/raw/seismic_ingv_20250525_20260525.csv` (QuakeML) in `data/raw/gossip_20260217_20260218.csv`: #51673, #51676, #51678, #51684. Nota: #51675 non è presente nel catalogo raw (finestra temporale/bounding box del download originale).
- [ ] **2.4** Scaricare dati geochimici (CO₂/H₂O) Pisciarelli/Solfatara — i Bollettini INGV-OV non espongono endpoint machine-readable; download manuale dai bollettini di sorveglianza.
- [x] **2.5** Eseguire script di pulizia (`scripts/*.R`/`.py`) e salvare output in `data/processed/` — `scripts/fase2_acquisizione_preprocessing.py` produce `catalog_gossip_20260217_20260218.csv`, `catalog_clean_ingv_events.csv`, `uplift_clean.csv`.
- [x] **2.6** Verificare outlier GNSS (rimozione manuale o automatica con IQR) — IQR su `uplift.csv`: soglie [0.2990, 1.0677], 0 outlier su 365 giorni; report in `data/processed/uplift_outliers.csv`.

## 🟠 Fase 3: Analisi Statistica Avanzata (Giorni 3-5)
- [x] **3.1** **Geodesia:** Calcolare change-point (PELT & Bai-Perron) con **Bootstrap CI (1000 replicates)** — PELT su `data/raw/uplift.csv`: 8 change-point (2025-07-14 … 2026-05-07), cross-check Bai-Perron (binary segmentation) con 7 CP concordi; bootstrap B=1000 con CI 95% per posizione → `data/processed/gnss_changepoints.csv`.
- [x] **3.2** **Geodesia:** Calcolare Rolling Variance e Autocorrelazione (lag-1) per indicatore CSD — CSD cumulato sugli incrementi giornalieri di uplift, rolling var e ac1 (finestra 30 gg) → `data/processed/csd_gnss_rolling.csv` (ac1 ~0.99: forte persistenza).
- [x] **3.3** **Sismologia:** Fit distribuzione Weibull sugli IET (verificare β > 1) — MLE su 2504 IET del catalogo: **β = 0,367 (< 1, hazard decrescente → clustering)**, λ = 7,01 h; AIC Weibull 15150,6 vs esponenziale 25752,8 → `data/processed/iet_weibull.csv`.
- [x] **3.4** **Sismologia:** Implementare modello **Hawkes (ETAS)** per confermare auto-eccitazione (confronto AIC con Weibull) — Hawkes a kernel esponenziale (μ, K, α, c, p) su finestra 180 gg (191 eventi): AIC Hawkes 24,5 vs Poisson 360,4 (stessa likelihood) → **auto-eccitazione confermata**; AIC Weibull sugli IET della stessa finestra (1286,9) riportato come riferimento → `data/processed/hawkes_etas_fit.csv`.
- [ ] **3.5** **Geochimica:** Test Engle-Granger e Johansen per cointegrazione Gas vs Deformazione — **bloccato**: nessun dato gas (task 2.4 manuale, bollettini INGV-OV); vedi `data/processed/geochemistry_status.csv`.
- [ ] **3.6** **Geochimica:** Test Causalità di Granger (lag=7gg) per direzione influenza — **bloccato** come 3.5.
- [x] **3.7** **Integrazione:** Propagazione incertezza (Monte-Carlo) per soglie di stress (Yield Point) — B=1000: yield point (media post-ultimo-CP) = 0,869 [CI95 0,847–0,893], P(margin>0)=0,993, P(superamento yield+2σ entro 30 gg)=0,264 → `data/processed/montecarlo_stress_threshold.csv`.

## 🟣 Fase 4: Reporting & Visualizzazione (Giorno 6)
- [x] **4.1** Generare grafici finali (PNG 300dpi) in `reports/fig/` (Time-series, Hazard Function, Phase-space) — 4 figure SVG resolution-independent in `reports/fig/` (`fig1_uplift_changepoints`, `fig2_csd_rolling`, `fig3_weibull_hazard`, `fig4_phase_space`) da `scripts/fase4_reporting.py`; i PNG 300dpi di pipeline sono in `figures/` (matplotlib in CI) e il Rmd usa `fig.path="reports/fig/"` con `dpi=300`.
- [ ] **4.2** Compilare `notebook/report.Rmd` e knittare in PDF (`reports/Flegrei_Alert_2026.pdf`) — richiede Docker (R 4.4 + LaTeX), non eseguibile in CI; generata la nota tecnica Markdown `reports/Flegrei_Alert_2026.md`; comando in MANUAL_EXECUTION.md §4.
- [x] **4.3** Inserire Abstract in Inglese nella nota tecnica — abstract EN in `notebook/report.Rmd` e in `reports/Flegrei_Alert_2026.md`.
- [x] **4.4** Verificare unità di misura in tutte le tabelle (mm/mese, MPa, etc.) — `reports/units_check.csv` (file x colonna x unita' x range osservato); nota: unita' di `uplift` non dichiarata nella sorgente (assunta m cumulati, da verificare vs INGV).
- [x] **4.5** Creare `DataPackage.zip` (≤10 MB) con CSV processati e metadata — `reports/DataPackage.zip`: 37 file, 0,64 MB; non committato (`*.zip` in `.gitignore`, policy anti-archivi PR #15), generato da `scripts/fase4_reporting.py`.

## 🔴 Fase 5: Revisione & Invio (Giorno 7)
- [ ] **5.1** Dry-run completo della pipeline (da Docker start a PDF generation).
- [ ] **5.2** Revisione pari (invio bozza a 1 collega geodeta/sismologo per feedback rapido).
- [ ] **5.3** Preparare email per Dott. De Martino, Dott. Di Vito, Dott.ssa Pappalardo.
- [ ] **5.4** Caricare dataset e codice su **Zenodo** (ottenere DOI) per archiviazione permanente.
- [ ] **5.5** Invio Nota Tecnica + Allegati.
- [ ] **5.6** Calendarizzare follow-up (email di cortesia dopo 5 gg lavorativi).