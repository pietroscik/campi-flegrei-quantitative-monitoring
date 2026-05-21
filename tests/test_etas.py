import pandas as pd

def test_lambda_positive():
    df = pd.read_csv("data/processed/etas_output.csv")
    assert (df["lambda_etas"] > 0).all()
