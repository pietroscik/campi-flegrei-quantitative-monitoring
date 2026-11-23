#!/usr/bin/env python3
"""
import_data.py - Pulizia e standardizzazione dei dati scaricati

Questo script legge i dati grezzi dalla cartella data/raw/ e produce
versioni pulite e standardizzate in data/processed/.
Supporta formati QuakeML (XML) per dati sismici INGV.
"""

import pandas as pd
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

def parse_quakeml(xml_file):
    """
    Parsa un file QuakeML INGV e restituisce un DataFrame con:
    - time: tempo origine
    - lat, lon, depth: coordinate ipocentrali
    - mag: magnitudo
    - location: descrizione località
    """
    ns = {
        'q': 'http://quakeml.org/xmlns/quakeml/1.2',
        'bed': 'http://quakeml.org/xmlns/bed/1.2',
        'ingv': 'http://webservices.ingv.it/'
    }
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    events = []
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID')
        
        # Descrizione (località)
        desc_elem = event.find('bed:description/bed:text', ns)
        location = desc_elem.text if desc_elem is not None else "Unknown"
        
        # Origine (tempo, coordinate)
        origin_ref = event.find('bed:preferredOriginID', ns)
        if origin_ref is not None:
            origin_id = origin_ref.text.split('=')[-1]
            # Cerca l'origine nel documento
            for origin in root.findall('.//bed:origin', ns):
                if origin.get('publicID') == origin_ref.text or origin.get('publicID', '').endswith(origin_id):
                    time_elem = origin.find('bed:time/bed:value', ns)
                    lat_elem = origin.find('bed:latitude/bed:value', ns)
                    lon_elem = origin.find('bed:longitude/bed:value', ns)
                    depth_elem = origin.find('bed:depth/bed:value', ns)
                    
                    origin_time = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00')) if time_elem is not None else None
                    lat = float(lat_elem.text) if lat_elem is not None else None
                    lon = float(lon_elem.text) if lon_elem is not None else None
                    depth = float(depth_elem.text) if depth_elem is not None else None
                    break
        else:
            origin_time = lat = lon = depth = None
        
        # Magnitudo
        mag_ref = event.find('bed:preferredMagnitudeID', ns)
        mag = None
        if mag_ref is not None:
            mag_id = mag_ref.text.split('=')[-1]
            for mag_elem in root.findall('.//bed:magnitude', ns):
                if mag_elem.get('publicID') == mag_ref.text or mag_elem.get('publicID', '').endswith(mag_id):
                    mag_val = mag_elem.find('bed:mag/bed:value', ns)
                    mag = float(mag_val.text) if mag_val is not None else None
                    break
        
        if origin_time and lat and lon and mag is not None:
            events.append({
                'time': origin_time,
                'lat': lat,
                'lon': lon,
                'depth': depth,
                'mag': mag,
                'location': location,
                'event_id': event_id
            })
    
    return pd.DataFrame(events)

def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # Assicura che la directory di output esista
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Importazione e pulizia dei dati...")
    
    # Processa file sismici (QuakeML XML)
    seismic_files = list(raw_dir.glob("*seismic*")) + list(raw_dir.glob("*SEISMIC*"))
    if seismic_files:
        for f in seismic_files:
            print(f"  Processing seismic file: {f.name}")
            try:
                # Controlla se è XML (QuakeML)
                with open(f, 'r') as check:
                    first_lines = check.read(200)
                
                if '<?xml' in first_lines or '<q:quakeml' in first_lines:
                    df = parse_quakeml(f)
                    output_name = f"seismic_clean_{f.stem}.csv"
                    df.to_csv(processed_dir / output_name, index=False)
                    print(f"    → Estratti {len(df)} eventi sismici → {output_name}")
                else:
                    print(f"    ⚠ Formato non riconosciuto: {f.name}")
            except Exception as e:
                print(f"    ⚠ Errore nel processing di {f.name}: {e}")
    
    # Processa file GNSS se esistono
    gnss_files = list(raw_dir.glob("*gnss*")) + list(raw_dir.glob("*GNSS*"))
    if gnss_files:
        for f in gnss_files:
            print(f"  Processing GNSS file: {f.name}")
            # Logica specifica per GNSS da implementare
    
    # Processa file geochimici se esistono
    gas_files = list(raw_dir.glob("*gas*")) + list(raw_dir.glob("*GAS*")) + list(raw_dir.glob("*chem*"))
    if gas_files:
        for f in gas_files:
            print(f"  Processing geochemical file: {f.name}")
            # Logica specifica per geochimica da implementare
    
    # Verifica output
    processed_files = list(processed_dir.glob("*.csv"))
    if processed_files:
        print(f"\n✅ Dati processati salvati in {processed_dir}/:")
        for pf in processed_files:
            print(f"   - {pf.name}")
    else:
        print("\n⚠ Nessun dato processato generato.")
    
    print("\nImportazione completata.")

if __name__ == "__main__":
    main()
