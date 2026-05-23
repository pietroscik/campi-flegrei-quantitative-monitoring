import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Endpoint configurabile tramite variabile d'ambiente
BASE_URL = os.getenv("INGV_API_URL", "https://webservices.ingv.it/fdsnws/event/1/query")

# Timeout configurabile
REQUEST_TIMEOUT = int(os.getenv("INGV_REQUEST_TIMEOUT", "30"))

# Validazione parametri di bounding box
VALID_LAT_RANGE = (-90, 90)
VALID_LON_RANGE = (-180, 180)


def validate_coordinate(value, coord_type="latitude"):
    """Valida che le coordinate siano entro i limiti validi"""
    if value is None:
        return True
    try:
        val = float(value)
        if coord_type == "latitude":
            return VALID_LAT_RANGE[0] <= val <= VALID_LAT_RANGE[1]
        else:
            return VALID_LON_RANGE[0] <= val <= VALID_LON_RANGE[1]
    except (ValueError, TypeError):
        return False


def fetch_ingv_events(starttime, endtime, minmag=0.0, maxlat=None, minlat=None,
                       maxlon=None, minlon=None):
    
    # Validazione input per prevenire injection
    try:
        if isinstance(starttime, datetime):
            start_str = starttime.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            start_str = str(starttime)
            # Validazione formato data
            pd.to_datetime(start_str)
        
        if isinstance(endtime, datetime):
            end_str = endtime.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            end_str = str(endtime)
            pd.to_datetime(end_str)
            
        # Validazione magnitudo
        minmag = float(minmag)
        if minmag < -2 or minmag > 15:
            raise ValueError("Magnitudo non valida")
            
    except Exception as e:
        raise ValueError(f"Parametri temporali o di magnitudo non validi: {e}")
    
    # Validazione coordinate
    for coord, ctype in [(maxlat, "latitude"), (minlat, "latitude"), 
                          (maxlon, "longitude"), (minlon, "longitude")]:
        if not validate_coordinate(coord, ctype):
            raise ValueError(f"Coordinata {ctype} non valida: {coord}")

    params = {
        "format": "geojson",
        "starttime": start_str,
        "endtime": end_str,
        "minmagnitude": minmag
    }

    # bounding box opzionale (Campi Flegrei approx)
    if all([maxlat, minlat, maxlon, minlon]):
        params.update({
            "maxlatitude": float(maxlat),
            "minlatitude": float(minlat),
            "maxlongitude": float(maxlon),
            "minlongitude": float(minlon)
        })

    try:
        response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Timeout nella richiesta INGV (> {REQUEST_TIMEOUT}s)")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Errore di connessione a INGV: {e}")

    if response.status_code == 204:
        return pd.DataFrame()

    if response.status_code != 200:
        # Non esporre dettagli sensibili dell'errore
        raise Exception(f"Errore nel recupero dati da INGV (status {response.status_code})")

    try:
        data = response.json()
    except ValueError:
        raise Exception("Formato risposta INGV non valido")

    features = data.get("features", [])

    rows = []
    for f in features:
        try:
            props = f["properties"]
            geom = f["geometry"]
            
            # Validazione struttura dati
            if "coordinates" not in geom or len(geom["coordinates"]) < 2:
                continue

            # handle time conversion properly - INGV returns ISO format string in 'time' property
            time_val = props.get("time")
            if time_val is not None:
                try:
                    time_val = pd.to_datetime(time_val, errors="coerce")
                except Exception:
                    time_val = None

            rows.append({
                "time": time_val,
                "magnitude": props.get("mag"),
                "depth": props.get("depth"),
                "longitude": geom["coordinates"][0],
                "latitude": geom["coordinates"][1],
                "place": props.get("place"),
                "event_id": props.get("eventId")
            })
        except (KeyError, IndexError, TypeError):
            # Salta feature malformate senza bloccare l'intero processo
            continue

    df = pd.DataFrame(rows)

    # sorting e cleaning base
    if not df.empty:
        df = df.sort_values("time")

    return df


def save_raw(df, path="data/raw/ingv_events.csv"):
    df.to_csv(path, index=False)
    print(f"[OK] Saved {len(df)} events -> {path}")


if __name__ == "__main__":

    end = datetime.utcnow()
    start = end - timedelta(days=365)  # ultimo anno (puoi cambiare)

    df = fetch_ingv_events(
        starttime=start,
        endtime=end,
        minmag=0.0,
        maxlat=41.2,
        minlat=40.7,
        maxlon=14.3,
        minlon=13.8
    )

    save_raw(df)
