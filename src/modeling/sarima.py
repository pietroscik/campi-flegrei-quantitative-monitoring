import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# SARIMA Model for Seismic Rate
# -----------------------------

def prepare_seismic_rate(df, freq="D"):
    """
    Convert catalog to seismic rate time series.
    
    Parameters:
    -----------
    df : DataFrame
        Seismic catalog with 'time' column
    freq : str
        Resampling frequency (e.g., 'D' for daily)
    
    Returns:
    --------
    rate : Series
        Seismic rate time series
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    
    rate = df.resample(freq).size()
    rate = rate.rename("seismic_rate")
    
    return rate


def fit_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Fit SARIMA model to time series.
    
    Parameters:
    -----------
    series : array-like
        Time series data
    order : tuple
        (p, d, q) ARIMA order
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    
    Returns:
    --------
    model : SARIMAXResults
        Fitted model
    """
    series = np.array(series)
    
    # Handle zeros by adding small constant
    if np.min(series) == 0:
        series = series + 0.1
    
    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        return model
    except Exception as e:
        print(f"[WARNING] SARIMA fitting failed: {e}")
        return None


def forecast_sarima(model, steps=30):
    """
    Generate forecasts from fitted SARIMA model.
    
    Parameters:
    -----------
    model : SARIMAXResults
        Fitted model
    steps : int
        Number of steps to forecast
    
    Returns:
    --------
    forecast : array
        Forecasted values
    conf_int : array
        Confidence intervals
    """
    if model is None:
        return None, None
    
    forecast = model.get_forecast(steps=steps)
    pred = forecast.predict_mean()
    conf_int = forecast.conf_int()
    
    return pred.values, conf_int.values


def run_sarima_analysis(input_path="data/processed/catalog_clean.csv", freq="D"):
    """
    Run SARIMA analysis on seismic rate.
    """
    df = pd.read_csv(input_path)
    
    # Prepare seismic rate
    rate = prepare_seismic_rate(df, freq=freq)
    
    # Fit SARIMA model
    model = fit_sarima(rate.values)
    
    if model is None:
        print("[ERROR] SARIMA model fitting failed")
        return None
    
    # Generate forecasts
    forecast, conf_int = forecast_sarima(model, steps=30)
    
    # Create results dataframe
    results = pd.DataFrame({
        "date": rate.index.tolist(),
        "observed": rate.values,
        "fitted": model.fittedvalues.values
    })
    
    # Add forecast
    if forecast is not None:
        last_date = results["date"].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=31, 
            freq=freq
        )[1:]
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast,
            "lower": conf_int[:, 0],
            "upper": conf_int[:, 1]
        })
        
        results = pd.concat([results, forecast_df], ignore_index=True)
    
    # Save results
    results.to_csv("data/processed/sarima_output.csv", index=False)
    
    print("[OK] SARIMA analysis saved -> data/processed/sarima_output.csv")
    print(f"[INFO] Model AIC: {model.aic:.2f}")
    print(f"[INFO] BIC: {model.bic:.2f}")
    
    return results


if __name__ == "__main__":
    run_sarima_analysis()