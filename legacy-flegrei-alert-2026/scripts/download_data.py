#!/usr/bin/env python3
"""
download_data.py - Download dati INGV reali per i Campi Flegrei

Questo script scarica i dati da fonti pubbliche certificate (INGV FDSN webservice)
e li salva nella cartella data/raw/. SOLO dati reali e validati.
"""

import requests
import os
from pathlib import Path
from datetime import datetime, timedelta

# Configurazione area Campi Flegrei
CF_BOUNDS = {
    "min_lat": 40.80,
    "max_lat": 40.95,
    "min_lon": 14.10,
    "max_lon": 14.25
}

def download_gnss_data():
    """
    Scarica dati GNSS reali dai server INGV o portali dedicati.
    
    Fonti:
    - INGV Osservatorio Vesuviano: http://www.ov.ingv.it/
    - RING Network: http://ring.gm.ingv.it/
    """
    print("Download dati GNSS reali...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # URL reali per dati GNSS
    gnss_urls = [
        "http://ring.gm.ingv.it/fdsnws/station/1/query?network=IV&station=RITE",
        "http://ring.gm.ingv.it/fdsnws/station/1/query?network=IV&station=BAIA"
    ]
    
    downloaded = False
    for url in gnss_urls:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                station_code = url.split("station=")[1].split("&")[0] if "station=" in url else "GNSS"
                output_file = raw_dir / f"gnss_{station_code}.txt"
                with open(output_file, 'w') as f:
                    f.write(response.text)
                print(f"  Scaricato: {output_file}")
                downloaded = True
        except Exception as e:
            print(f"  Warning: Impossibile scaricare da {url}: {e}")
    
    if not downloaded:
        print("\n  AZIONE RICHIESTA:")
        print("  1. Visitare http://ring.gm.ingv.it/ o http://www.ov.ingv.it/")
        print("  2. Scaricare dati GNSS per l'area dei Campi Flegrei")
        print("  3. Salvare i file in data/raw/ con prefisso 'gnss_'")
        print("  Il sistema richiede dati GNSS reali per l'analisi multi-segnale.")

def download_seismic_data(days_back=365):
    """
    Scarica dati sismici reali dal catalogo INGV FDSN.
    
    Utilizza il webservice ufficiale INGV per ottenere cataloghi sismici
    validati per l'area dei Campi Flegrei.
    
    Parameters:
        days_back (int): Giorni da recuperare (default: 365)
    """
    print(f"Download dati Sismici reali (ultimi {days_back} giorni)...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # INGV FDSN Webservice URL
    base_url = "https://webservices.ingv.it/fdsnws/event/1/query"
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    
    params = {
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minlat": CF_BOUNDS["min_lat"],
        "maxlat": CF_BOUNDS["max_lat"],
        "minlon": CF_BOUNDS["min_lon"],
        "maxlon": CF_BOUNDS["max_lon"],
        "minmag": 0.0,
        "format": "csv"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code == 200 and len(response.text) > 100:
            output_file = raw_dir / f"seismic_ingv_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"  Scaricato: {output_file}")
            print(f"  Periodo: {start_time.strftime('%Y-%m-%d')} al {end_time.strftime('%Y-%m-%d')}")
            
            # Conta eventi (escludendo header)
            lines = response.text.strip().split('\n')
            n_events = len(lines) - 5  # Sottrai righe di header INGV
            print(f"  Eventi trovati: ~{n_events}")
        else:
            print(f"  Warning: Nessuno scaricato (status: {response.status_code})")
            print("  Verificare i parametri di ricerca o la disponibilità del servizio.")
            
    except Exception as e:
        print(f"  Errore nel download: {e}")
        print("\n  AZIONE RICHIESTA:")
        print("  1. Visitare http://iside.rm.ingv.it/iside/standard/index.jsp")
        print("  2. Definire area: Lat 40.80-40.95, Lon 14.10-14.25")
        print("  3. Scaricare catalogo CSV")
        print(f"  4. Salvare in {raw_dir} con prefisso 'seismic_'")

def download_gas_data():
    """
    Scarica dati geochimici reali dalle stazioni di monitoraggio INGV.
    
    Fonti:
    - INGV Osservatorio Vesuviano: misure CO2, SO2, temperature
    - Portale dati geochimici: http://www.ov.ingv.it/geochimica
    """
    print("Download dati Geochimici reali...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder per URL reali (da configurare con endpoint specifici)
    gas_sources = [
        "Solfatara - CO2/SO2 ratio",
        "Pisciarelli - Temperature fumarole",
        "Agnano - Misure ground gas"
    ]
    
    print("\n  Fonti dati geochimici disponibili:")
    for source in gas_sources:
        print(f"    - {source}")
    
    print("\n  AZIONE RICHIESTA:")
    print("  1. Visitare http://www.ov.ingv.it/geochimica o contattare INGV-OV")
    print("  2. Richiedere dati geochimici per Campi Flegrei")
    print("  3. Salvare file CSV in data/raw/ con prefisso 'gas_' o 'geochem_'")
    print("  Nota: Alcuni dati geochimici potrebbero richiedere autorizzazione.")

def main():
    print("=" * 60)
    print("Flegrei Alert 2026 - Download Dati Reali e Validati")
    print("=" * 60)
    print("\nIMPORTANTE: Questo sistema utilizza SOLO dati reali e validati.")
    print("Nessun dato sintetico verrà generato o utilizzato.\n")
    
    download_seismic_data(days_back=365)
    print()
    download_gnss_data()
    print()
    download_gas_data()
    
    print("\n" + "=" * 60)
    print("Download completato. Verificare i file in data/raw/")
    print("=" * 60)
    print("\nNota: Se alcuni download non sono riusciti, seguire le istruzioni")
    print("per il download manuale dai portali indicati.")

if __name__ == "__main__":
    main()
