import pandas as pd
import numpy as np

# Bounding box Campi Flegrei (più stretto e realistico)
CF_BBOX = {
    "min_lat": 40.75,
    "max_lat": 40.95,
    "min_lon": 14.05,
    "max_lon": 14.25
}

def load_catalog(path="data/raw/ingv_events.csv"):
    df = pd.read_csv(path)
    return df


def clean_time(df):
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    return df


def clean_magnitude(df):
    df = df.dropna(subset=["magnitude"])
    df = df[df["magnitude"] >= 0]   # elimina valori negativi errati
    return df


def spatial_filter(df):
    df = df[
        (df["latitude"] >= CF_BBOX["min_lat"]) &
        (df["latitude"] <= CF_BBOX["max_lat"]) &
        (df["longitude"] >= CF_BBOX["min_lon"]) &
        (df["longitude"] <= CF_BBOX["max_lon"])
    ]
    return df


def remove_duplicates(df):
    df = df.drop_duplicates(subset=["event_id"])
    return df


def sort_catalog(df):
    return df.sort_values("time").reset_index(drop=True)


def feature_engineering(df):
    # feature base per analisi futura
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.date

    # energia sismica proxy (scala semplificata)
    df["energy_proxy"] = 10 ** (1.5 * df["magnitude"])

    return df


def build_catalog(raw_path, output_path="data/processed/catalog_clean.csv"):

    df = load_catalog(raw_path)

    df = clean_time(df)
    df = clean_magnitude(df)
    df = spatial_filter(df)
    df = remove_duplicates(df)
    df = sort_catalog(df)
    df = feature_engineering(df)

    df.to_csv(output_path, index=False)

    print(f"[OK] Clean catalog saved: {output_path}")
    print(f"[INFO] Events retained: {len(df)}")

    return df


if __name__ == "__main__":
    build_catalog("data/raw/ingv_events.csv")
