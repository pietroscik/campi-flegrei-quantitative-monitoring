from fastapi import FastAPI, HTTPException
import pandas as pd
import time
import os
from pathlib import Path

app = FastAPI(title="CF Observatory API")

# Configurazione sicura dei percorsi
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Whitelist dei file consentiti
ALLOWED_FILES = {
    "status": "early_warning_system.csv",
    "unrest": "unrest_index.csv",
    "etas": "etas_output.csv"
}

def load_csv_safely(filename: str) -> pd.DataFrame:
    """Carica CSV solo da percorsi whitelistati e validati"""
    # Verifica che il filename sia nella whitelist
    if filename not in ALLOWED_FILES.values():
        raise HTTPException(status_code=403, detail="Accesso al file non consentito")
    
    filepath = DATA_DIR / filename
    
    # Previene path traversal risolvendo il percorso assoluto
    try:
        resolved_path = filepath.resolve()
        if not str(resolved_path).startswith(str(DATA_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Percorso non valido")
    except Exception:
        raise HTTPException(status_code=400, detail="Errore nel percorso del file")
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="File non trovato")
    
    try:
        return pd.read_csv(resolved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Errore nella lettura del file")


@app.get("/status")
def status():
    try:
        df = load_csv_safely("early_warning_system.csv")
        if df.empty:
            raise HTTPException(status_code=404, detail="Nessun dato disponibile")
        latest = df.tail(1).to_dict(orient="records")[0]
        return latest
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno del server")


@app.get("/unrest")
def unrest():
    try:
        df = load_csv_safely("unrest_index.csv")
        return df.tail(200).to_dict(orient="records")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno del server")


@app.get("/etas")
def etas():
    try:
        df = load_csv_safely("etas_output.csv")
        return df.tail(200).to_dict(orient="records")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Errore interno del server")
