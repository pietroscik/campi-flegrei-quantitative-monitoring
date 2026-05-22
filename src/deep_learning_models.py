"""
Deep Learning Models for Hybrid Seismic Monitoring Framework.

This module implements:
1. LSTM Network: For non-linear time-series forecasting of seismic rates.
2. Variational Autoencoder (VAE): For unsupervised anomaly detection.
3. Data Generator: Robust sequence creation for training without data leakage.

Author: Campi Flegrei Monitoring Team
Date: 2024
"""

import os
import warnings

# Silenzia i warning hardware e di logica deprecata (es. tf.placeholder)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, TimeDistributed, 
        BatchNormalization, LayerNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Using scikit-learn fallback models.")
    
    # Define dummy types for type hints when TensorFlow is not available
    class Model:
        pass
    
    # Import scikit-learn as fallback
    try:
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.ensemble import IsolationForest
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False


class SequenceGenerator:
    """
    Data generator for time series sequences.
    
    Creates supervised learning samples from time series data
    without data leakage by maintaining temporal order.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data (1D or 2D)
    lookback : int
        Number of time steps to look back for prediction
    horizon : int
        Number of time steps to predict ahead
    batch_size : int
        Batch size for training
    shuffle : bool
        Whether to shuffle sequences (default: False for time series)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 30,
        horizon: int = 7,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for SequenceGenerator")
        
        self.data = np.asarray(data)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate valid indices
        self.valid_indices = len(self.data) - lookback - horizon + 1
        if self.valid_indices <= 0:
            raise ValueError(
                f"Insufficient data: {len(self.data)} samples with "
                f"lookback={lookback} and horizon={horizon}"
            )
        
        self.indices = np.arange(self.valid_indices)
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of sequences."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            X_batch.append(self.data[i:i + self.lookback])
            y_batch.append(self.data[i + self.lookback:i + self.lookback + self.horizon])
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        """Shuffle indices at epoch end if shuffle=True."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_lstm_model(
    input_shape: Tuple[int, int],
    horizon: int = 7,
    lstm_units: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> Model:
    """
    Build LSTM model for seismic rate forecasting.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data (lookback, n_features)
    horizon : int
        Prediction horizon
    lstm_units : list
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns
    -------
    keras.Model
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to build LSTM model")
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Stack LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = LSTM(units, return_sequences=return_sequences)(x)
        x = LayerNormalization()(x)
        if return_sequences:
            x = Dropout(dropout_rate)(x)
    
    # Dense output layer
    outputs = Dense(horizon)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_autoencoder(
    input_dim: int,
    latent_dim: int = 8,
    hidden_dims: List[int] = [32, 16],
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Build Variational Autoencoder for anomaly detection.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    latent_dim : int
        Dimension of latent space
    hidden_dims : list
        Dimensions of hidden layers
    learning_rate : float
        Learning rate for optimizer
    
    Returns
    -------
    tuple
        (autoencoder_model, encoder_model)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to build Autoencoder")
    
    # Encoder
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    for dim in hidden_dims:
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
    
    # Latent space (mean and log-variance for VAE)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    
    # Add KL divergence loss using a custom layer (Keras 3 compatibility)
    class KLDivergenceLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            z_m, z_l_v = inputs
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_l_v - tf.square(z_m) - tf.exp(z_l_v), axis=1)
            )
            self.add_loss(0.1 * kl_loss)
            return z_m
            
    KLDivergenceLayer()([z_mean, z_log_var])
    
    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    x = z
    for dim in reversed(hidden_dims):
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
    
    outputs = Dense(input_dim)(x)
    
    autoencoder = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs=inputs, outputs=z_mean)
    
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse' # Reconstruction loss
    )
    
    return autoencoder, encoder


class LSTMForecaster:
    """
    LSTM-based forecaster for seismic time series.
    
    When TensorFlow is not available, falls back to MLP (Multi-Layer Perceptron)
    from scikit-learn for non-linear forecasting.
    
    Parameters
    ----------
    lookback : int
        Number of past time steps to use
    horizon : int
        Number of future time steps to predict
    lstm_units : list
        Units in LSTM layers (ignored if using sklearn fallback)
    """
    
    def __init__(
        self,
        lookback: int = 30,
        horizon: int = 7,
        lstm_units: List[int] = [64, 32]
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.lstm_units = lstm_units
        self.model = None
        self.history = None
        self.scaler = None
        
        if not TENSORFLOW_AVAILABLE and not SKLEARN_AVAILABLE:
            raise ImportError(
                "Either TensorFlow or scikit-learn is required for LSTMForecaster"
            )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model (LSTM if TF available, MLP if sklearn fallback).
        
        Parameters
        ----------
        X_train : np.ndarray
            Training sequences (n_samples, lookback, n_features)
        y_train : np.ndarray
            Training targets (n_samples, horizon)
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation targets
        epochs : int
            Maximum number of iterations (for sklearn)
        batch_size : int
            Batch size (ignored for sklearn)
        verbose : int
            Verbosity level
        
        Returns
        -------
        dict
            Training history
        """
        # Flatten input for sklearn compatibility
        n_samples = X_train.shape[0]
        X_flat = X_train.reshape(n_samples, -1)
        
        if TENSORFLOW_AVAILABLE:
            # Use TensorFlow LSTM
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            self.model = build_lstm_model(
                input_shape=input_shape,
                horizon=self.horizon,
                lstm_units=self.lstm_units
            )
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=15,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            validation_data = (X_val, y_val) if X_val is not None else None
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            ).history
        else:
            # Use sklearn MLP as fallback
            if verbose:
                print("   Using scikit-learn MLP (TensorFlow not available)")
            
            self.scaler = StandardScaler()
            y_scaled = self.scaler.fit_transform(y_train)
            
            # Set validation_fraction only if validation data is provided
            val_frac = 0.1 if X_val is None and len(X_train) > 50 else None
            
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=epochs,
                early_stopping=True if X_val is None else False,
                validation_fraction=val_frac if val_frac is not None else 0.1,
                verbose=verbose > 0,
                random_state=42
            )
            
            self.model.fit(X_flat, y_scaled)
            
            # Create pseudo-history for compatibility
            self.history = {'loss': [self.model.loss_]}
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences (n_samples, lookback, n_features)
        
        Returns
        -------
        np.ndarray
            Predictions (n_samples, horizon)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        if TENSORFLOW_AVAILABLE:
            return self.model.predict(X, verbose=0)
        else:
            pred = self.model.predict(X_flat)
            if self.scaler is not None:
                pred = self.scaler.inverse_transform(pred)
            return pred
    
    def forecast_recursive(
        self,
        initial_sequence: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Generate multi-step forecast using recursive strategy.
        
        Parameters
        ----------
        initial_sequence : np.ndarray
            Initial sequence (lookback, n_features)
        steps : int
            Number of steps to forecast
        
        Returns
        -------
        np.ndarray
            Forecasted values (steps,)
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        current_seq = initial_sequence.copy()
        forecasts = []
        
        for _ in range(steps):
            # Reshape for model input
            X_input = current_seq[-self.lookback:].reshape(1, self.lookback, -1)
            
            # Predict next horizon
            pred = self.predict(X_input)[0]
            
            # Take first prediction
            forecasts.append(pred[0])
            
            # Update sequence (shift and add new prediction)
            new_row = np.zeros(current_seq.shape[1])
            new_row[0] = pred[0]  # Assuming single feature
            current_seq = np.vstack([current_seq[1:], new_row])
        
        return np.array(forecasts)


class AnomalyDetector:
    """
    Autoencoder-based anomaly detector for seismic features.
    
    When TensorFlow is not available, falls back to sklearn's Isolation Forest
    for unsupervised anomaly detection.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent space (ignored if using sklearn fallback)
    threshold_percentile : float
        Percentile for anomaly threshold (e.g., 95)
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        threshold_percentile: float = 95.0
    ):
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.autoencoder = None
        self.encoder = None
        self.threshold = None
        self.mean_error = None
        self.std_error = None
        self.isolation_forest = None
        
        # Check for available backends
        if not TENSORFLOW_AVAILABLE and not SKLEARN_AVAILABLE:
            raise ImportError(
                "Either TensorFlow or scikit-learn is required for AnomalyDetector"
            )
    
    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the anomaly detector on normal data.
        
        When TensorFlow is available, trains a Variational Autoencoder.
        When using sklearn fallback, trains an Isolation Forest.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data (n_samples, n_features)
        epochs : int
            Number of training epochs (ignored for sklearn)
        batch_size : int
            Batch size (ignored for sklearn)
        verbose : int
            Verbosity level
        """
        if TENSORFLOW_AVAILABLE:
            # Use TensorFlow Autoencoder
            input_dim = X_train.shape[1]
            
            self.autoencoder, self.encoder = build_autoencoder(
                input_dim=input_dim,
                latent_dim=self.latent_dim
            )
            
            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            self.autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Compute reconstruction errors on training data
            reconstructions = self.autoencoder.predict(X_train, verbose=0)
            errors = np.mean(np.square(X_train - reconstructions), axis=1)
            
            self.mean_error = np.mean(errors)
            self.std_error = np.std(errors)
            self.threshold = np.percentile(errors, self.threshold_percentile)
        else:
            # Use sklearn Isolation Forest as fallback
            if verbose:
                print("   Using scikit-learn Isolation Forest (TensorFlow not available)")
            
            # Calculate contamination from threshold_percentile
            contamination = 1.0 - (self.threshold_percentile / 100.0)
            
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=min(100, len(X_train)),
                verbose=verbose > 0
            )
            
            self.isolation_forest.fit(X_train)
            
            # Compute anomaly scores on training data
            scores = -self.isolation_forest.score_samples(X_train)  # Higher = more anomalous
            
            self.mean_error = np.mean(scores)
            self.std_error = np.std(scores)
            self.threshold = np.percentile(scores, self.threshold_percentile)
    
    def detect(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies.
        
        When using autoencoder: based on reconstruction error.
        When using Isolation Forest: based on anomaly score.
        
        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, n_features)
        
        Returns
        -------
        tuple
            (anomaly_flags, reconstruction_errors, z_scores)
        """
        if TENSORFLOW_AVAILABLE:
            if self.autoencoder is None:
                raise ValueError("Model must be trained before detection")
            
            reconstructions = self.autoencoder.predict(X, verbose=0)
            errors = np.mean(np.square(X - reconstructions), axis=1)
        else:
            if self.isolation_forest is None:
                raise ValueError("Model must be trained before detection")
            
            # Get anomaly scores (negative = more anomalous in sklearn, so we negate)
            errors = -self.isolation_forest.score_samples(X)
        
        # Z-score of anomaly score/error
        z_scores = (errors - self.mean_error) / (self.std_error + 1e-8)
        
        # Binary anomaly flags
        anomaly_flags = errors > self.threshold
        
        return anomaly_flags, errors, z_scores
    
    def get_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores (normalized reconstruction error).
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Anomaly scores (higher = more anomalous)
        """
        _, errors, z_scores = self.detect(X)
        return z_scores


def prepare_seismic_features(
    catalog: np.ndarray,
    window_size: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for deep learning from seismic catalog.
    
    Parameters
    ----------
    catalog : np.ndarray
        Seismic catalog with columns: [time, magnitude, depth, ...]
    window_size : int
        Window size for aggregating features
    
    Returns
    -------
    tuple
        (features, timestamps)
    """
    from scipy.stats import binned_statistic
    
    times = catalog[:, 0]
    magnitudes = catalog[:, 1]
    depths = catalog[:, 2] if catalog.shape[1] > 2 else np.zeros_like(magnitudes)
    
    # Create time bins
    t_start = times.min()
    t_end = times.max()
    n_bins = int((t_end - t_start) / window_size) + 1
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    
    # Aggregate features per window
    features = []
    timestamps = []
    
    for i in range(n_bins):
        mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        
        window_mags = magnitudes[mask]
        window_depths = depths[mask]
        
        # Feature vector
        feat = [
            mask.sum(),  # Event count
            window_mags.mean(),  # Mean magnitude
            window_mags.std() if len(window_mags) > 1 else 0,  # Mag std
            window_mags.max(),  # Max magnitude
            window_depths.mean(),  # Mean depth
            window_depths.std() if len(window_depths) > 1 else 0,  # Depth std
            np.sum(10 ** (1.5 * window_mags)),  # Cumulative energy (proxy)
        ]
        
        features.append(feat)
        timestamps.append(bin_edges[i])
    
    return np.array(features), np.array(timestamps)


def run_dl_pipeline(
    catalog_path: str = "data/processed/catalog_clean.csv",
    output_dir: str = "data/processed"
) -> Optional[pd.DataFrame]:
    """
    Esegue la pipeline di Deep Learning per il forecasting della sismicità 
    e il rilevamento delle anomalie.
    """
    import os
    from datetime import timedelta
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(catalog_path):
        print(f"ERROR: Catalog not found at {catalog_path}")
        return None
        
    df = pd.read_csv(catalog_path, parse_dates=['time'])
    df_clean = df.dropna(subset=['time', 'magnitude']).copy()
    df_clean = df_clean.set_index('time').sort_index()
    
    # Aggregazione del tasso sismico giornaliero
    daily_rate = df_clean.resample('D').size()
    if len(daily_rate) == 0:
        return None
        
    full_date_range = pd.date_range(start=daily_rate.index.min(), end=daily_rate.index.max(), freq='D')
    daily_rate = daily_rate.reindex(full_date_range, fill_value=0)
    
    seismic_data = daily_rate.values.astype(float)
    if len(seismic_data) < 100:
        print("Not enough data for Deep Learning (requires >= 100 days). Skipping DL step.")
        return None
        
    # Normalizzazione per la stabilità della rete neurale
    mean_val = seismic_data.mean()
    std_val = seismic_data.std()
    normalized_data = (seismic_data - mean_val) / (std_val + 1e-8)
    
    # 1. LSTM Forecasting (Previsione dei prossimi 7 giorni)
    lookback = 30
    horizon = 7
    print(f"  Training LSTM forecaster on {len(seismic_data)} days of historical data...")
    
    # Creazione delle sequenze di addestramento su tutto il dataset
    X_train, y_train = [], []
    for i in range(len(normalized_data) - lookback - horizon + 1):
        X_train.append(normalized_data[i:i+lookback].reshape(-1, 1))
        y_train.append(normalized_data[i+lookback:i+lookback+horizon])
        
    if len(X_train) > 0:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        forecaster = LSTMForecaster(lookback=lookback, horizon=horizon, lstm_units=[64, 32])
        forecaster.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        
        # Previsione del futuro
        last_sequence = normalized_data[-lookback:].reshape(1, lookback, 1)
        forecast_norm = forecaster.predict(last_sequence)[0]
        forecast_real = forecast_norm * std_val + mean_val
        forecast_real = np.maximum(0, forecast_real)  # Previene i tassi sismici negativi
        
        future_dates = [daily_rate.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
        forecast_df = pd.DataFrame({'time': future_dates, 'forecasted_rate': forecast_real})
        forecast_path = os.path.join(output_dir, "dl_forecast.csv")
        forecast_df.to_csv(forecast_path, index=False)
        print(f"  [OK] 7-Day LSTM Forecast saved -> {forecast_path}")
        
    # 2. VAE Anomaly Detection (Unsupervised Autoencoder)
    print("  Training Autoencoder Anomaly Detector...")
    features = []
    window = 7
    for i in range(len(seismic_data) - window + 1):
        w_data = seismic_data[i:i+window]
        features.append([np.mean(w_data), np.std(w_data), np.max(w_data),
                         np.min(w_data), np.median(w_data), w_data[-1] - w_data[0],
                         len(w_data[w_data > np.mean(w_data)])])
    features = np.array(features)
    
    if len(features) > 0:
        detector = AnomalyDetector(latent_dim=4, threshold_percentile=95)
        detector.fit(features, epochs=30, batch_size=16, verbose=0)
        flags, errors, z_scores = detector.detect(features)
        
        # Allineamento della lunghezza per combaciare con l'indice temporale
        pad_len = len(daily_rate) - len(flags)
        pad_flags = np.pad(flags, (pad_len, 0), constant_values=False)
        pad_scores = np.pad(z_scores, (pad_len, 0), constant_values=0.0)
        
        anomaly_df = pd.DataFrame({
            'time': daily_rate.index,
            'dl_anomaly_score': pad_scores,
            'dl_is_anomaly': pad_flags.astype(int)
        })
        anomaly_path = os.path.join(output_dir, "dl_anomalies.csv")
        anomaly_df.to_csv(anomaly_path, index=False)
        print(f"  [OK] DL Anomalies saved -> {anomaly_path}")
        
    return forecast_df


if __name__ == "__main__":
    """
    Test Deep Learning Models on REAL INGV seismic data.
    
    This script validates the LSTM forecaster and Autoencoder anomaly detector
    using actual seismic catalog from Campi Flegrei (INGV).
    
    NO SYNTHETIC DATA IS USED. All results are reproducible with real observations.
    """
    import os
    
    print("=" * 70)
    print("HYBRID AI-STATISTICAL FRAMEWORK - VALIDATION ON REAL DATA")
    print("=" * 70)
    
    if not TENSORFLOW_AVAILABLE and not SKLEARN_AVAILABLE:
        print("ERROR: Neither TensorFlow nor scikit-learn available.")
        print("Install with: pip install tensorflow  OR  pip install scikit-learn")
        exit(1)
    elif not TENSORFLOW_AVAILABLE:
        print("NOTE: Using scikit-learn MLP as fallback (TensorFlow not available)")
    
    # Load REAL INGV seismic catalog
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "data", 
        "raw", 
        "ingv_events.csv"
    )
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"ERROR: INGV catalog not found at {data_path}")
        print("Please run: python src/ingestion/fetch_ingv.py")
        exit(1)
    
    print(f"\nLoading real seismic data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['time'])
    print(f"Loaded {len(df)} real earthquakes from Campi Flegrei (INGV)")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
    
    # Prepare daily seismic rate time series
    print("\nPreparing daily seismic rate time series...")
    df_clean = df.dropna(subset=['time', 'magnitude']).copy()
    df_clean = df_clean.set_index('time').sort_index()
    
    # Aggregate to daily counts
    daily_rate = df_clean.resample('D').size()
    
    # Fill missing days with zeros
    full_date_range = pd.date_range(
        start=daily_rate.index.min(),
        end=daily_rate.index.max(),
        freq='D'
    )
    daily_rate = daily_rate.reindex(full_date_range, fill_value=0)
    
    print(f"Daily rate time series: {len(daily_rate)} days")
    print(f"Mean daily rate: {daily_rate.mean():.2f} events/day")
    print(f"Max daily rate: {daily_rate.max()} events/day")
    
    # Convert to numpy array
    seismic_data = daily_rate.values.astype(float)
    
    # Check minimum data requirements
    min_required = 100  # Minimum days for training
    if len(seismic_data) < min_required:
        print(f"\nWARNING: Insufficient data ({len(seismic_data)} days < {min_required})")
        print("Deep learning models require at least 100 days of observations.")
        print("Skipping DL validation.")
        exit(0)
    
    # Split data: 80% train, 20% test (temporal order preserved)
    split_idx = int(len(seismic_data) * 0.8)
    train_data = seismic_data[:split_idx]
    test_data = seismic_data[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} days ({train_data.sum():.0f} total events)")
    print(f"  Testing: {len(test_data)} days ({test_data.sum():.0f} total events)")
    
    # Normalize data for DL
    mean_val = train_data.mean()
    std_val = train_data.std()
    train_normalized = (train_data - mean_val) / (std_val + 1e-8)
    test_normalized = (test_data - mean_val) / (std_val + 1e-8)
    
    # Create sequences for LSTM
    lookback = 30
    horizon = 7
    
    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i:i+lookback].reshape(-1, 1))
            y.append(data[i+lookback:i+lookback+horizon])
        return np.array(X), np.array(y)
    
    print(f"\nCreating sequences (lookback={lookback}, horizon={horizon})...")
    X_train, y_train = create_sequences(train_normalized, lookback, horizon)
    X_test, y_test = create_sequences(test_normalized, lookback, horizon)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # =====================================================================
    # TEST 1: LSTM FORECASTER ON REAL DATA
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: LSTM FORECASTER - REAL SEISMIC DATA")
    print("=" * 70)
    
    forecaster = LSTMForecaster(lookback=lookback, horizon=horizon, lstm_units=[64, 32])
    
    print("\nTraining LSTM on real Campi Flegrei seismicity...")
    history = forecaster.fit(
        X_train, y_train,
        X_val=X_test[:10], y_val=y_test[:10],  # Small validation set
        epochs=50,
        batch_size=16,
        verbose=1
    )
    
    # Predict on test set
    print("\nEvaluating on test set...")
    predictions = forecaster.predict(X_test)
    
    # Calculate RMSE on normalized data
    rmse_normalized = np.sqrt(np.mean((predictions - y_test) ** 2))
    
    # Denormalize for interpretation
    predictions_denorm = predictions * std_val + mean_val
    y_test_denorm = y_test * std_val + mean_val
    rmse_real = np.sqrt(np.mean((predictions_denorm - y_test_denorm) ** 2))
    
    print(f"\nLSTM Performance on REAL data:")
    print(f"  RMSE (normalized): {rmse_normalized:.4f}")
    print(f"  RMSE (real scale): {rmse_real:.4f} events/day")
    print(f"  MAE (real scale): {np.mean(np.abs(predictions_denorm - y_test_denorm)):.4f} events/day")
    
    # Sample predictions
    print("\nSample predictions (first 5 test sequences):")
    for i in range(min(5, len(predictions))):
        print(f"  Seq {i+1}: True={y_test_denorm[i, 0]:.2f}, Pred={predictions_denorm[i, 0]:.2f}")
    
    # =====================================================================
    # TEST 2: AUTOENCODER ANOMALY DETECTOR ON REAL DATA
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: AUTOENCODER ANOMALY DETECTOR - REAL SEISMIC DATA")
    print("=" * 70)
    
    # Prepare features for anomaly detection
    # Use sliding window statistics as features
    def compute_features(data, window=7):
        features = []
        for i in range(len(data) - window + 1):
            window_data = data[i:i+window]
            feat = [
                np.mean(window_data),
                np.std(window_data),
                np.max(window_data),
                np.min(window_data),
                np.median(window_data),
                window_data[-1] - window_data[0],  # Trend
                len(window_data[window_data > window_data.mean()])  # Count above mean
            ]
            features.append(feat)
        return np.array(features)
    
    print("\nComputing seismic features for anomaly detection...")
    train_features = compute_features(train_data, window=7)
    test_features = compute_features(test_data, window=7)
    
    print(f"Feature dimensions: {train_features.shape[1]} features")
    print(f"Training samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")
    
    # Train autoencoder on NORMAL data (first 70% of training set)
    normal_cutoff = int(len(train_features) * 0.7)
    X_normal = train_features[:normal_cutoff]
    
    print(f"\nTraining autoencoder on {len(X_normal)} normal samples...")
    detector = AnomalyDetector(latent_dim=4, threshold_percentile=95)
    detector.fit(X_normal, epochs=50, batch_size=16, verbose=1)
    
    # Detect anomalies in test set
    print("\nDetecting anomalies in test period...")
    flags, errors, z_scores = detector.detect(test_features)
    
    n_anomalies = flags.sum()
    anomaly_rate = n_anomalies / len(flags) * 100
    
    print(f"\nAnomaly Detection Results on REAL data:")
    print(f"  Total test samples: {len(test_features)}")
    print(f"  Anomalies detected: {n_anomalies} ({anomaly_rate:.1f}%)")
    print(f"  Mean reconstruction error: {errors.mean():.4f}")
    print(f"  Max reconstruction error: {errors.max():.4f}")
    print(f"  Threshold (95th percentile): {detector.threshold:.4f}")
    
    # Show top anomalies
    if n_anomalies > 0:
        top_indices = np.argsort(errors)[-5:][::-1]
        print("\nTop 5 anomalous periods:")
        for idx in top_indices:
            date_idx = split_idx + idx  # Offset by split
            if date_idx < len(daily_rate):
                date = daily_rate.index[date_idx]
                print(f"  {date.strftime('%Y-%m-%d')}: Error={errors[idx]:.4f}, "
                      f"Z-score={z_scores[idx]:.2f}, "
                      f"Actual rate={test_data[idx]:.0f} events/day")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE - REAL INGV DATA")
    print("=" * 70)
    print(f"Dataset: Campi Flegrei seismic catalog (INGV)")
    print(f"Period: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
    print(f"Total events: {len(df)}")
    print(f"\nLSTM Forecasting:")
    print(f"  RMSE: {rmse_real:.4f} events/day")
    print(f"  Successfully trained on {len(train_data)} days of real observations")
    print(f"\nAutoencoder Anomaly Detection:")
    print(f"  Detected {n_anomalies} anomalous periods in test set")
    print(f"  Trained on {len(X_normal)} normal seismic sequences")
    print("\n✓ All models validated on REAL seismic data (NO synthetic data used)")
    print("=" * 70)
