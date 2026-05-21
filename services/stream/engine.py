import time
import pandas as pd

def stream_state(interval=10):

    while True:

        df = pd.read_csv("../data/processed/early_warning_system.csv")
        latest = df.tail(1).to_dict(orient="records")[0]

        print("[LIVE STATE]", latest)

        time.sleep(interval)
