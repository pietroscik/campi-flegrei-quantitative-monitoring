from src.ingestion.fetch_ingv import fetch_ingv_events, save_raw
from src.preprocessing.clean_catalog import build_catalog
from src.analysis.b_value import run_b_analysis
from src.analysis.anomaly_bvalue import run_anomaly
from src.analysis.multi_signal_model import run_multisignal
from src.analysis.early_warning import run_warning_system
from src.analysis.etas_mle import run_mle

from datetime import datetime, timedelta


def pipeline():

    print("\n[PIPELINE START]\n")

    # 1. INGESTION
    end = datetime.utcnow()
    start = end - timedelta(days=365)

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

    # 2. CLEANING
    clean_df = build_catalog("data/raw/ingv_events.csv")

    # 3. B-VALUE
    run_b_analysis()

    # 4. ANOMALY
    run_anomaly()

    # 5. MULTI-SIGNAL
    run_multisignal()

    # 6. EARLY WARNING
    run_warning_system()

    # 7. ETAS MLE
    run_mle()

    print("\n[PIPELINE COMPLETE]\n")


if __name__ == "__main__":
    pipeline()
