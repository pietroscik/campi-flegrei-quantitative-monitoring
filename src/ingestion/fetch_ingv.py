import requests
import pandas as pd
from datetime import datetime, timedelta

# Endpoint ufficiale FDSN IRIS/INGV-like (event service)
BASE_URL = "https://webservices.ingv.it/fdsnws/event/1/query"

def fetch_ingv_events(starttime, endtime, minmag=0.0, maxlat=None, minlat=None,
                       maxlon=None, minlon=None):

    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmag
    }

    # bounding box opzionale (Campi Flegrei approx)
    if maxlat and minlat and maxlon and minlon:
        params.update({
            "maxlatitude": maxlat,
            "minlatitude": minlat,
            "maxlongitude": maxlon,
            "minlongitude": minlon
        })

    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code != 200:
        raise Exception(f"INGV request failed: {response.status_code}")

    data = response.json()

    features = data.get("features", [])

    rows = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]

        # handle time conversion properly
        time_val = props.get("time")
        if time_val is not None:
            time_val = pd.to_datetime(time_val, unit="ms", errors="coerce")

        rows.append({
            "time": time_val,
            "magnitude": props.get("mag"),
            "depth": props.get("depth"),
            "longitude": geom["coordinates"][0],
            "latitude": geom["coordinates"][1],
            "place": props.get("place"),
            "event_id": props.get("eventId")
        })

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
        starttime=start.isoformat(),
        endtime=end.isoformat(),
        minmag=0.0,
        maxlat=41.2,
        minlat=40.7,
        maxlon=14.3,
        minlon=13.8
    )

    save_raw(df)
