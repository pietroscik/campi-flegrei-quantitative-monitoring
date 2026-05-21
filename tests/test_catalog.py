import pandas as pd
import pytest
import os

@pytest.mark.skipif(not os.path.exists("data/processed/catalog_clean.csv"), reason="Data not generated")
def test_catalog_not_empty():
    df = pd.read_csv("data/processed/catalog_clean.csv")
    assert len(df) > 0

@pytest.mark.skipif(not os.path.exists("data/processed/catalog_clean.csv"), reason="Data not generated")
def test_no_nan_magnitude():
    df = pd.read_csv("data/processed/catalog_clean.csv")
    assert df["magnitude"].isna().sum() == 0
