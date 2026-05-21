from fetch_ingv import fetch_ingv_events, save_raw
from datetime import datetime, timedelta

def run_update():

    end = datetime.utcnow()
    start = end - timedelta(days=30)  # aggiornamento incrementale

    df = fetch_ingv_events(
        starttime=start.isoformat(),
        endtime=end.isoformat(),
        minmag=0.0,
        maxlat=41.2,
        minlat=40.7,
        maxlon=14.3,
        minlon=13.8
    )

    save_raw(df, "data/raw/ingv_events_latest.csv")


if _name_ == "_main_":
    run_update()
