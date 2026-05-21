from core.ingestion import fetch_ingv_events
from core.bvalue import run_b_analysis
from core.etas import run_etas
from core.unrest import run_multisignal
from core.warning import run_warning_system

from datetime import datetime, timedelta

def cycle():

    end = datetime.utcnow()
    start = end - timedelta(days=7)

    df = fetch_ingv_events(
        starttime=start.isoformat(),
        endtime=end.isoformat(),
        minmag=0.0
    )

    run_b_analysis()
    run_etas()
    run_multisignal()
    run_warning_system()

    print("✔ cycle completed")


if __name__ == "__main__":
    cycle()
