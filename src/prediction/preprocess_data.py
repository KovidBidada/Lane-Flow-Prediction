"""
Data Preprocessing Module
Prepares METR-LA and DETRAC data for prediction models
"""

import sys
import os
# Make project root importable so running this file directly works
# (adds the project root two levels up: Focus/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Local helper functions
# These provide the small utilities your script expects.
# If you have a utils_pred module with equivalent functions, you can remove/replace these.
# ---------------------------

def save_pickle(obj, path: str):
    """Save object to pickle file (ensures parent dir exists)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {path}")


def load_pickle(path: str):
    """Load object from pickle file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def fill_missing_values(arr: np.ndarray, method: str = "linear") -> np.ndarray:
    """
    Fill missing values in a 1D array.
    Supports 'linear' (interpolation) and 'ffill' / 'bfill'.
    """
    a = np.array(arr, dtype=float)
    if np.isnan(a).all():
        return np.zeros_like(a)

    # Use pandas Series interpolation for robustness
    s = pd.Series(a)
    if method == "linear":
        s = s.interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        s = s.fillna(method="ffill").fillna(method="bfill")
    elif method == "bfill":
        s = s.fillna(method="bfill").fillna(method="ffill")
    else:
        s = s.fillna(0)

    # any remaining NaN -> 0
    s = s.fillna(0)
    return s.values


def normalize_data(data: np.ndarray, scaler_type: str = "standard") -> Tuple[np.ndarray, object]:
    """
    Normalize 2D data (T, features).
    Returns normalized_data, scaler_object.
    scaler_type currently supports 'standard' only.
    """
    if scaler_type != "standard":
        logger.warning("Only 'standard' scaler supported. Falling back to StandardScaler.")

    shape = data.shape
    flat = data.reshape(-1, shape[-1]) if data.ndim == 2 else data.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat)
    normalized = scaled.reshape(shape)
    return normalized, scaler


def split_train_val_test(data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Split 2D time-series data into train/val/test contiguous chunks.
    Returns (train, val, test) arrays.
    """
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


def create_sequences(data: np.ndarray, seq_len: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences.
    Input:
      data: (T, features)
    Output:
      X: (N, seq_len, features)
      y: (N, pred_horizon, features)
    """
    T = data.shape[0]
    X, y = [], []
    for i in range(T - seq_len - pred_horizon + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + pred_horizon])
    if len(X) == 0:
        return np.zeros((0, seq_len, data.shape[1])), np.zeros((0, pred_horizon, data.shape[1]))
    return np.array(X), np.array(y)


# ---------------------------
# End local helpers
# ---------------------------


class DataPreprocessor:
    """Preprocess traffic data for prediction models"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor"""
        # Resolve config path relative to project root if needed
        config_path_resolved = config_path
        if not Path(config_path_resolved).exists():
            # Try to find config in project root (two levels up)
            project_root = Path(__file__).resolve().parents[2]
            candidate = project_root / config_path
            if candidate.exists():
                config_path_resolved = str(candidate)

        with open(config_path_resolved, 'r') as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config.get('paths', {})
        self.pred_config = self.config.get('prediction', {})

        logger.info("DataPreprocessor initialized")

    def load_metr_la_csv(self) -> pd.DataFrame:
        """Load METR-LA CSV data"""
        csv_path = self.paths.get('metr_la_csv', 'data/raw/METR-LA.csv')

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"METR-LA CSV not found: {csv_path}")

        logger.info(f"Loading METR-LA data from {csv_path}")
        df = pd.read_csv(csv_path)

        return df

    def load_metr_la_adjacency(self) -> Optional[np.ndarray]:
        """Load METR-LA adjacency matrix"""
        adj_path = self.paths.get('metr_la_adj', 'data/raw/adj_METR-LA.pkl')

        if not Path(adj_path).exists():
            logger.warning(f"Adjacency matrix not found: {adj_path}")
            return None

        logger.info(f"Loading adjacency matrix from {adj_path}")
        with open(adj_path, 'rb') as f:
            # different pickles may need different encodings
            try:
                adj_data = pickle.load(f)
            except Exception:
                f.seek(0)
                adj_data = pickle.load(f, encoding='latin1')

        # Extract adjacency matrix (format may vary)
        if isinstance(adj_data, tuple):
            adj_matrix = adj_data[0]
        elif isinstance(adj_data, dict):
            adj_matrix = adj_data.get('adj_mx', adj_data)
        else:
            adj_matrix = adj_data

        return adj_matrix

    def preprocess_metr_la(self, fill_missing: bool = True) -> np.ndarray:
        """
        Preprocess METR-LA dataset

        Args:
            fill_missing: Whether to fill missing values

        Returns:
            data: Preprocessed data (timesteps, sensors)
        """
        df = self.load_metr_la_csv()

        # Assuming CSV format: first column is timestamp, rest are sensor readings
        if 'timestamp' in df.columns or df.columns[0].lower() in ['time', 'date']:
            timestamps = pd.to_datetime(df.iloc[:, 0])
            data = df.iloc[:, 1:].values.astype(float)
        else:
            timestamps = None
            data = df.values.astype(float)

        logger.info(f"Loaded data shape: {data.shape}")

        # Handle missing values
        if fill_missing:
            logger.info("Filling missing values...")
            for i in range(data.shape[1]):
                data[:, i] = fill_missing_values(data[:, i], method='linear')

        # Remove any remaining NaN rows
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]

        logger.info(f"After cleaning: {data.shape}")

        return data

    def create_train_test_split(self, data: np.ndarray):
        """
        Create train/validation/test split and sequences

        Args:
            data: Input data (timesteps, features)

        Returns:
            Dictionary with train/val/test data
        """
        # Normalize data
        normalized_data, scaler = normalize_data(data, scaler_type='standard')

        # Split data
        train_data, val_data, test_data = split_train_val_test(
            normalized_data,
            train_ratio=0.7,
            val_ratio=0.15
        )

        logger.info(f"Split sizes - Train: {len(train_data)}, "
                    f"Val: {len(val_data)}, Test: {len(test_data)}")

        # Create sequences
        seq_length = int(self.pred_config.get('sequence_length', 12))
        pred_horizon = int(self.pred_config.get('prediction_horizon', 3))

        X_train, y_train = create_sequences(train_data, seq_length, pred_horizon)
        X_val, y_val = create_sequences(val_data, seq_length, pred_horizon)
        X_test, y_test = create_sequences(test_data, seq_length, pred_horizon)

        logger.info(f"Sequence shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'raw_data': data
        }

    def save_preprocessed_data(self, data_dict: dict):
        """Save preprocessed data"""
        output_path = self.paths.get('metr_la_prepared', 'data/processed/metr_la_prepared.npy')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.save(output_path, data_dict['raw_data'])
        logger.info(f"Saved preprocessed data to {output_path}")

        # Save scaler
        scaler_path = self.paths.get('scaler', 'data/processed/scaler.pkl')
        save_pickle(data_dict['scaler'], scaler_path)

        # Save train/test splits
        split_path = Path(output_path).parent / "train_test_split.npz"
        np.savez(
            split_path,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test']
        )
        logger.info(f"Saved train/test splits to {split_path}")

    def load_preprocessed_data(self):
        """Load preprocessed data"""
        prepared_path = Path(self.paths.get('metr_la_prepared', 'data/processed/metr_la_prepared.npy'))
        split_path = prepared_path.parent / "train_test_split.npz"

        if not split_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {split_path}")

        data = np.load(split_path, allow_pickle=True)
        scaler = load_pickle(self.paths.get('scaler', 'data/processed/scaler.pkl'))

        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'scaler': scaler
        }

    def preprocess_detrac_data(self, detections_list: List[List[Dict]]) -> np.ndarray:
        """
        Preprocess DETRAC detection data

        Args:
            detections_list: List of detections per frame

        Returns:
            processed: Processed detection features
        """
        features = []

        for frame_detections in detections_list:
            # Extract features from detections
            num_vehicles = len(frame_detections)

            # Calculate average confidence
            if num_vehicles > 0:
                avg_confidence = np.mean([d.get('confidence', 0) for d in frame_detections])

                # Calculate spatial distribution
                centers = np.array([d.get('center', (0, 0)) for d in frame_detections])
                if centers.shape[0] > 0 and centers.ndim == 2:
                    spatial_std_x = np.std(centers[:, 0]) if num_vehicles > 1 else 0
                    spatial_std_y = np.std(centers[:, 1]) if num_vehicles > 1 else 0
                else:
                    spatial_std_x = 0
                    spatial_std_y = 0

                # Calculate average bbox size
                bbox_sizes = []
                for d in frame_detections:
                    bbox = d.get('bbox', None)
                    if bbox and len(bbox) == 4:
                        bbox_sizes.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                avg_size = np.mean(bbox_sizes) if bbox_sizes else 0
            else:
                avg_confidence = 0
                spatial_std_x = 0
                spatial_std_y = 0
                avg_size = 0

            frame_features = [
                num_vehicles,
                avg_confidence,
                spatial_std_x,
                spatial_std_y,
                avg_size
            ]

            features.append(frame_features)

        features = np.array(features)
        logger.info(f"DETRAC features shape: {features.shape}")

        return features

    def generate_synthetic_data(self, num_samples: int = 10000,
                                num_sensors: int = 207) -> np.ndarray:
        """
        Generate synthetic traffic data for testing

        Args:
            num_samples: Number of time steps
            num_sensors: Number of sensors

        Returns:
            synthetic_data: Generated traffic data
        """
        logger.info(f"Generating synthetic data: {num_samples} x {num_sensors}")

        # Create time-varying patterns
        t = np.linspace(0, 100, num_samples)

        data = np.zeros((num_samples, num_sensors))

        for i in range(num_sensors):
            # Base traffic pattern with daily cycles
            daily_cycle = 50 * (1 + np.sin(2 * np.pi * t / 24))

            # Add weekly pattern
            weekly_cycle = 20 * np.sin(2 * np.pi * t / (24 * 7))

            # Add random noise
            noise = np.random.randn(num_samples) * 5

            # Combine patterns
            data[:, i] = daily_cycle + weekly_cycle + noise + 30

            # Ensure non-negative
            data[:, i] = np.maximum(data[:, i], 0)

        return data


def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()

    try:
        # Try to load real METR-LA data
        logger.info("Attempting to load METR-LA data...")
        data = preprocessor.preprocess_metr_la()
    except FileNotFoundError:
        # Generate synthetic data if real data not available
        logger.warning("METR-LA data not found. Generating synthetic data...")
        data = preprocessor.generate_synthetic_data()

    # Create train/test split
    data_dict = preprocessor.create_train_test_split(data)

    # Save preprocessed data
    preprocessor.save_preprocessed_data(data_dict)

    # Print statistics
    print("\n" + "=" * 50)
    print("Data Preprocessing Complete")
    print("=" * 50)
    print(f"Training samples: {len(data_dict['X_train'])}")
    print(f"Validation samples: {len(data_dict['X_val'])}")
    print(f"Test samples: {len(data_dict['X_test'])}")
    print(f"Sequence length: {data_dict['X_train'].shape[1]}")
    print(f"Number of features: {data_dict['X_train'].shape[2]}")
    print(f"Prediction horizon: {data_dict['y_train'].shape[1]}")
    print("=" * 50)


if __name__ == "__main__":
    # Allow running as module: python -m src.prediction.preprocess_data
    main()
