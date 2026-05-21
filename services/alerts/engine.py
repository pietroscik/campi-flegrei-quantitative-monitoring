import pandas as pd

def check_alerts():

    df = pd.read_csv("../data/processed/early_warning_system.csv")

    latest = df.tail(1).iloc[0]

    if latest["alert_flag"] == 1:
        print("🚨 CRITICAL ALERT TRIGGERED")
        print(latest.to_dict())
