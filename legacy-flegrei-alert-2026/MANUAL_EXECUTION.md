# Guida all'Esecuzione Manuale

Questa guida descrive come eseguire i singoli step della pipeline utilizzando direttamente i comandi Docker da terminale (Git Bash, PowerShell o CMD).

> **Nota:** Assicurati di eseguire questi comandi dalla radice del progetto.

## 1. Costruzione dell'Immagine Docker

Prima di eseguire qualsiasi analisi, è necessario costruire l'immagine che contiene R, Python e tutte le librerie necessarie.

```bash
docker build -t flegrei-alert .
```

## 2. Acquisizione Dati

### Opzione A: Download Automatico (Web)
Scarica i dati sismici dai servizi INGV.

```bash
# Git Bash / Linux / Mac
docker run --rm -v "$PWD:/app" flegrei-alert python3 scripts/download_data.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" flegrei-alert python3 scripts/download_data.py
```

### Opzione B: Importazione Dati Locali
Se hai file CSV locali in `data/raw/` (es. `gnss_raw.csv`), esegui lo script di pulizia:

```bash
# Git Bash
docker run --rm -v "$PWD:/app" flegrei-alert python3 scripts/import_data.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" flegrei-alert python3 scripts/import_data.py
```

## 3. Analisi Statistica

Esegue in sequenza:
1.  Analisi GNSS (Change-Point Detection)
2.  Analisi Sismica (Modelli Hawkes/ETAS)
3.  Analisi Geochimica (Causalità di Granger)

```bash
# Git Bash
docker run --rm -v "$PWD:/app" flegrei-alert sh -c 'Rscript scripts/gnss_analysis.R && python3 scripts/seismic_hawkes.py && Rscript scripts/geo_chemistry.R'

# Windows PowerShell
docker run --rm -v "${PWD}:/app" flegrei-alert sh -c 'Rscript scripts/gnss_analysis.R && python3 scripts/seismic_hawkes.py && Rscript scripts/geo_chemistry.R'
```

## 4. Generazione Report PDF

Compila il notebook RMarkdown finale utilizzando l'ambiente Docker (garantisce che R e LaTeX siano configurati correttamente).

```bash
# Git Bash
docker run --rm -v "$PWD:/app" flegrei-alert Rscript -e "rmarkdown::render('notebook/report.Rmd')"

# Windows PowerShell
docker run --rm -v "${PWD}:/app" flegrei-alert Rscript -e "rmarkdown::render('notebook/report.Rmd')"
```

Il report finale sarà disponibile in: `notebook/report.pdf` (o `reports/` se configurato nello script).