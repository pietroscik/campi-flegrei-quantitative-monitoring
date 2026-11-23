# 📋 Checklist Operativa - Nota Tecnica INGV (Feb 2026)

## 🟢 Fase 1: Setup & Riproducibilità (Giorni 1-2)
- [ ] **1.1** Inizializzare repository Git (`git init`) e creare repo remoto (GitHub/GitLab privato).
- [ ] **1.2** Creare `Dockerfile` con R 4.4, Python 3.12 e pacchetti necessari (`tidyverse`, `survival`, `lmtest`, `tick`).
- [ ] **1.3** Testare build Docker (`docker build -t flegrei-alert .`).
- [ ] **1.4** Configurare `.gitignore` per escludere dati sensibili o file >100MB (usare Git LFS se necessario).
- [ ] **1.5** Scrivere `metadata.yaml` con descrizione fonti dati (INGV Open Data) e licenza (CC-BY-4.0).

## 🔵 Fase 2: Acquisizione & Preprocessing Dati (Giorni 2-3)
- [ ] **2.1** Scaricare serie GNSS RITE (BEPOS) da portale INGV e salvare in `data/raw/`.
- [ ] **2.2** Calcolare checksum SHA-256 dei file scaricati e registrarli in `metadata.yaml`.
- [ ] **2.3** Estrarre catalogo sismico (GOSSIP) 17-18 Feb 2026 (Eventi #51675, #51678, etc.).
- [ ] **2.4** Scaricare dati geochimici (CO₂/H₂O) Pisciarelli/Solfatara.
- [ ] **2.5** Eseguire script di pulizia (`scripts/*.R`/`.py`) e salvare output in `data/processed/`.
- [ ] **2.6** Verificare outlier GNSS (rimozione manuale o automatica con IQR).

## 🟠 Fase 3: Analisi Statistica Avanzata (Giorni 3-5)
- [ ] **3.1** **Geodesia:** Calcolare change-point (PELT & Bai-Perron) con **Bootstrap CI (1000 replicates)**.
- [ ] **3.2** **Geodesia:** Calcolare Rolling Variance e Autocorrelazione (lag-1) per indicatore CSD.
- [ ] **3.3** **Sismologia:** Fit distribuzione Weibull sugli IET (verificare β > 1).
- [ ] **3.4** **Sismologia:** Implementare modello **Hawkes (ETAS)** per confermare auto-eccitazione (confronto AIC con Weibull).
- [ ] **3.5** **Geochimica:** Test Engle-Granger e Johansen per cointegrazione Gas vs Deformazione.
- [ ] **3.6** **Geochimica:** Test Causalità di Granger (lag=7gg) per direzione influenza.
- [ ] **3.7** **Integrazione:** Propagazione incertezza (Monte-Carlo) per soglie di stress (Yield Point).

## 🟣 Fase 4: Reporting & Visualizzazione (Giorno 6)
- [ ] **4.1** Generare grafici finali (PNG 300dpi) in `reports/fig/` (Time-series, Hazard Function, Phase-space).
- [ ] **4.2** Compilare `notebook/report.Rmd` e knittare in PDF (`reports/Flegrei_Alert_2026.pdf`).
- [ ] **4.3** Inserire Abstract in Inglese nella nota tecnica.
- [ ] **4.4** Verificare unità di misura in tutte le tabelle (mm/mese, MPa, etc.).
- [ ] **4.5** Creare `DataPackage.zip` (≤10 MB) con CSV processati e metadata.

## 🔴 Fase 5: Revisione & Invio (Giorno 7)
- [ ] **5.1** Dry-run completo della pipeline (da Docker start a PDF generation).
- [ ] **5.2** Revisione pari (invio bozza a 1 collega geodeta/sismologo per feedback rapido).
- [ ] **5.3** Preparare email per Dott. De Martino, Dott. Di Vito, Dott.ssa Pappalardo.
- [ ] **5.4** Caricare dataset e codice su **Zenodo** (ottenere DOI) per archiviazione permanente.
- [ ] **5.5** Invio Nota Tecnica + Allegati.
- [ ] **5.6** Calendarizzare follow-up (email di cortesia dopo 5 gg lavorativi).