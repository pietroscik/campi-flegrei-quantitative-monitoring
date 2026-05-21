import pandas as pd

def test_catalog_not_empty():
    df = pd.read_csv("data/processed/catalog_clean.csv")
    assert len(df) > 0


def test_no_nan_magnitude():
    df = pd.read_csv("data/processed/catalog_clean.csv")
    assert df["magnitude"].isna().sum() == 0
