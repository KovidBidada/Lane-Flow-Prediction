"""
Utility functions for traffic prediction
"""

import numpy as np
import pickle
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pickle(file_path: str):
    """Load pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path: str):
    """Save data to pickle file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {file_path}")


def smooth_series(series: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing
    
    Args:
        series: Time series data
        window_size: Window size for smoothing
        
    Returns:
        smoothed: Smoothed series
    """
    if len(series) < window_size:
        return series
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(series, kernel, mode='same')
    
    return smoothed


def detect_anomalies(series: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect anomalies using z-score
    
    Args:
        series: Time series data
        threshold: Z-score threshold
        
    Returns:
        anomalies: Boolean array indicating anomalies
    """
    mean = np.mean(series)
    std = np.std(series)
    
    z_scores = np.abs((series - mean) / (std + 1e-10))
    anomalies = z_scores > threshold
    
    return anomalies


def fill_missing_values(series: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Fill missing values in time series
    
    Args:
        series: Time series with NaN values
        method: Interpolation method ('linear', 'forward', 'backward')
        
    Returns:
        filled: Series with filled values
    """
    filled = series.copy()
    
    if method == 'linear':
        # Linear interpolation
        nans = np.isnan(filled)
        if nans.any():
            indices = np.arange(len(filled))
            filled[nans] = np.interp(
                indices[nans],
                indices[~nans],
                filled[~nans]
            )
    
    elif method == 'forward':
        # Forward fill
        mask = np.isnan(filled)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = filled[idx]
    
    elif method == 'backward':
        # Backward fill
        filled = filled[::-1]
        mask = np.isnan(filled)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = filled[idx][::-1]
    
    return filled


def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix between features
    
    Args:
        data: Input data (samples, features)
        
    Returns:
        corr_matrix: Correlation matrix
    """
    return np.corrcoef(data, rowvar=False)


def create_sequences(data: np.ndarray, seq_length: int, pred_horizon: int):
    """
    Create sequences for time series prediction
    
    Args:
        data: Time series data (timesteps, features)
        seq_length: Length of input sequence
        pred_horizon: Number of steps to predict
        
    Returns:
        X: Input sequences (samples, seq_length, features)
        y: Target sequences (samples, pred_horizon, features)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_horizon])
    
    return np.array(X), np.array(y)


def normalize_data(data: np.ndarray, scaler_type: str = 'standard'):
    """
    Normalize data
    
    Args:
        data: Input data
        scaler_type: Type of scaler ('standard' or 'minmax')
        
    Returns:
        normalized_data: Normalized data
        scaler: Fitted scaler object
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Reshape if needed
    original_shape = data.shape
    if len(original_shape) > 2:
        data_2d = data.reshape(-1, original_shape[-1])
        normalized = scaler.fit_transform(data_2d)
        normalized = normalized.reshape(original_shape)
    else:
        normalized = scaler.fit_transform(data)
    
    return normalized, scaler


def denormalize_data(data: np.ndarray, scaler):
    """
    Denormalize data using fitted scaler
    
    Args:
        data: Normalized data
        scaler: Fitted scaler object
        
    Returns:
        denormalized: Original scale data
    """
    original_shape = data.shape
    if len(original_shape) > 2:
        data_2d = data.reshape(-1, original_shape[-1])
        denormalized = scaler.inverse_transform(data_2d)
        denormalized = denormalized.reshape(original_shape)
    else:
        denormalized = scaler.inverse_transform(data)
    
    return denormalized


def split_train_val_test(data: np.ndarray, train_ratio: float = 0.7,
                          val_ratio: float = 0.15):
    """
    Split data into train, validation, and test sets
    
    Args:
        data: Input data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        train, val, test: Split datasets
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    return train, val, test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate prediction metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        metrics: Dictionary of metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def add_time_features(timestamps: np.ndarray) -> np.ndarray:
    """
    Extract time-based features from timestamps
    
    Args:
        timestamps: Array of timestamp strings or datetime objects
        
    Returns:
        features: Time-based features (hour, day_of_week, is_weekend)
    """
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps)
    elif not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    
    features = np.stack([
        timestamps.hour,
        timestamps.dayofweek,
        (timestamps.dayofweek >= 5).astype(int),  # is_weekend
        timestamps.day,
        timestamps.month
    ], axis=1)
    
    return features


def get_peak_hours(data: np.ndarray, timestamps: np.ndarray, 
                   top_n: int = 3) -> list:
    """
    Identify peak traffic hours
    
    Args:
        data: Traffic data
        timestamps: Corresponding timestamps
        top_n: Number of peak hours to return
        
    Returns:
        peak_hours: List of peak hour indices
    """
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps)
    
    hourly_avg = {}
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if hour not in hourly_avg:
            hourly_avg[hour] = []
        hourly_avg[hour].append(np.mean(data[i]))
    
    # Calculate average for each hour
    hour_means = {h: np.mean(vals) for h, vals in hourly_avg.items()}
    
    # Get top N hours
    sorted_hours = sorted(hour_means.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [h for h, _ in sorted_hours[:top_n]]
    
    return peak_hours


def exponential_smoothing(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential smoothing
    
    Args:
        series: Time series data
        alpha: Smoothing factor (0 < alpha < 1)
        
    Returns:
        smoothed: Exponentially smoothed series
    """
    smoothed = np.zeros_like(series)
    smoothed[0] = series[0]
    
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
    
    return smoothed


def detect_trends(series: np.ndarray, window: int = 10) -> str:
    """
    Detect trend in time series
    
    Args:
        series: Time series data
        window: Window size for trend detection
        
    Returns:
        trend: 'increasing', 'decreasing', or 'stable'
    """
    if len(series) < window:
        return 'stable'
    
    recent = series[-window:]
    slope = np.polyfit(np.arange(window), recent, 1)[0]
    
    threshold = np.std(series) * 0.1
    
    if slope > threshold:
        return 'increasing'
    elif slope < -threshold:
        return 'decreasing'
    else:
        return 'stable'


def aggregate_by_time(data: np.ndarray, timestamps: np.ndarray,
                      freq: str = 'H') -> tuple:
    """
    Aggregate data by time frequency
    
    Args:
        data: Traffic data
        timestamps: Corresponding timestamps
        freq: Frequency ('H' for hourly, 'D' for daily, 'W' for weekly)
        
    Returns:
        aggregated_data, aggregated_timestamps
    """
    df = pd.DataFrame({
        'value': data.flatten() if len(data.shape) > 1 else data,
        'timestamp': pd.to_datetime(timestamps)
    })
    
    df.set_index('timestamp', inplace=True)
    aggregated = df.resample(freq).mean()
    
    return aggregated['value'].values, aggregated.index.values


def calculate_congestion_index(traffic_flow: np.ndarray,
                               capacity: float = 1000.0) -> np.ndarray:
    """
    Calculate congestion index (0-1 scale)
    
    Args:
        traffic_flow: Traffic flow values
        capacity: Road capacity
        
    Returns:
        congestion_index: Congestion values (0=free flow, 1=jammed)
    """
    congestion = traffic_flow / capacity
    congestion = np.clip(congestion, 0, 1)
    
    return congestion


if __name__ == "__main__":
    # Test utilities
    print("Testing traffic prediction utilities...")
    
    # Create sample data
    data = np.random.randn(1000, 5)
    
    # Test sequence creation
    X, y = create_sequences(data, seq_length=12, pred_horizon=3)
    print(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
    
    # Test normalization
    normalized, scaler = normalize_data(data)
    print(f"Data normalized: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # Test denormalization
    denormalized = denormalize_data(normalized, scaler)
    print(f"Denormalization error: {np.abs(data - denormalized).mean():.6f}")
    
    # Test metrics
    y_true = np.random.randn(100, 5)
    y_pred = y_true + np.random.randn(100, 5) * 0.1
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    print("All tests passed!")