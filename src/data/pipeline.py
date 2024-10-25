"""
Data processing pipeline for traffic congestion prediction.
Handles data ingestion, cleaning, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

from .schema import (
    TrafficData, WeatherData, EventData, Location,
    RoadType, WeatherCondition, EventType, CongestionLevel,
    TimeFeatures, DataValidator, create_feature_dataframe
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Pipeline for ingesting and processing traffic data."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data ingestion pipeline."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
        self.validator = DataValidator()
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def parse_traffic_data(self, df: pd.DataFrame) -> List[TrafficData]:
        """Parse DataFrame into TrafficData objects."""
        traffic_data = []
        
        for _, row in df.iterrows():
            try:
                location = Location(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    address=row.get('address'),
                    city=row.get('city'),
                    state=row.get('state')
                )
                
                traffic = TrafficData(
                    timestamp=pd.to_datetime(row['timestamp']),
                    location=location,
                    road_id=str(row['road_id']),
                    road_type=RoadType(row['road_type']),
                    traffic_volume=int(row['traffic_volume']),
                    average_speed=float(row['average_speed']),
                    free_flow_speed=float(row['free_flow_speed']),
                    occupancy_rate=float(row['occupancy_rate'])
                )
                
                if self.validator.validate_traffic_data(traffic):
                    traffic_data.append(traffic)
                else:
                    logger.warning(f"Invalid traffic data at row {_}")
                    
            except Exception as e:
                logger.warning(f"Error parsing traffic data at row {_}: {e}")
                continue
        
        logger.info(f"Parsed {len(traffic_data)} valid traffic records")
        return traffic_data
    
    def parse_weather_data(self, df: pd.DataFrame) -> List[WeatherData]:
        """Parse DataFrame into WeatherData objects."""
        weather_data = []
        
        for _, row in df.iterrows():
            try:
                location = Location(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude'])
                )
                
                weather = WeatherData(
                    timestamp=pd.to_datetime(row['timestamp']),
                    location=location,
                    temperature=float(row['temperature']),
                    humidity=float(row['humidity']),
                    precipitation=float(row['precipitation']),
                    wind_speed=float(row['wind_speed']),
                    visibility=float(row['visibility']),
                    condition=WeatherCondition(row['condition'])
                )
                
                if self.validator.validate_weather_data(weather):
                    weather_data.append(weather)
                else:
                    logger.warning(f"Invalid weather data at row {_}")
                    
            except Exception as e:
                logger.warning(f"Error parsing weather data at row {_}: {e}")
                continue
        
        logger.info(f"Parsed {len(weather_data)} valid weather records")
        return weather_data
    
    def parse_event_data(self, df: pd.DataFrame) -> List[EventData]:
        """Parse DataFrame into EventData objects."""
        event_data = []
        
        for _, row in df.iterrows():
            try:
                location = Location(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude'])
                )
                
                event = EventData(
                    event_id=str(row['event_id']),
                    timestamp=pd.to_datetime(row['timestamp']),
                    location=location,
                    event_type=EventType(row['event_type']),
                    expected_attendance=row.get('expected_attendance'),
                    duration_hours=row.get('duration_hours'),
                    impact_radius_km=float(row.get('impact_radius_km', 5.0))
                )
                
                if self.validator.validate_event_data(event):
                    event_data.append(event)
                else:
                    logger.warning(f"Invalid event data at row {_}")
                    
            except Exception as e:
                logger.warning(f"Error parsing event data at row {_}: {e}")
                continue
        
        logger.info(f"Parsed {len(event_data)} valid event records")
        return event_data


class FeatureEngineer:
    """Feature engineering for traffic prediction."""
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features for time series data."""
        df_with_lags = df.copy()
        
        for col in columns:
            for lag in lags:
                df_with_lags[f"{col}_lag_{lag}"] = df_with_lags[col].shift(lag)
        
        return df_with_lags
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling window features."""
        df_with_rolling = df.copy()
        
        for col in columns:
            for window in windows:
                df_with_rolling[f"{col}_rolling_mean_{window}"] = (
                    df_with_rolling[col].rolling(window=window).mean()
                )
                df_with_rolling[f"{col}_rolling_std_{window}"] = (
                    df_with_rolling[col].rolling(window=window).std()
                )
                df_with_rolling[f"{col}_rolling_max_{window}"] = (
                    df_with_rolling[col].rolling(window=window).max()
                )
                df_with_rolling[f"{col}_rolling_min_{window}"] = (
                    df_with_rolling[col].rolling(window=window).min()
                )
        
        return df_with_rolling
    
    @staticmethod
    def create_traffic_density_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create traffic density and flow features."""
        df_enhanced = df.copy()
        
        df_enhanced['traffic_density'] = df_enhanced['traffic_volume'] / 10  # Assuming 10km segments
        
        df_enhanced['speed_deviation'] = (
            df_enhanced['free_flow_speed'] - df_enhanced['average_speed']
        )
        
        df_enhanced['congestion_index'] = (
            df_enhanced['occupancy_rate'] / 100 * (1 - df_enhanced['speed_ratio'])
        )
        
        df_enhanced['traffic_efficiency'] = (
            df_enhanced['traffic_volume'] * df_enhanced['speed_ratio']
        )
        
        return df_enhanced
    
    @staticmethod
    def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features based on location."""
        df_spatial = df.copy()
        
        center_lat = df_spatial['latitude'].mean()
        center_lon = df_spatial['longitude'].mean()
        
        df_spatial['distance_from_center'] = np.sqrt(
            (df_spatial['latitude'] - center_lat) ** 2 +
            (df_spatial['longitude'] - center_lon) ** 2
        ) * 111  # Convert to approximate km
        
        df_spatial['lat_grid'] = (df_spatial['latitude'] * 100).astype(int)
        df_spatial['lon_grid'] = (df_spatial['longitude'] * 100).astype(int)
        df_spatial['spatial_cluster'] = (
            df_spatial['lat_grid'].astype(str) + '_' + df_spatial['lon_grid'].astype(str)
        )
        
        return df_spatial


class DataPreprocessor:
    """Preprocesses data for machine learning models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'congestion_level'
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        df_processed = df.copy()
        
        df_processed = df_processed.sort_values(['road_id', 'timestamp'])
        
        feature_engineer = FeatureEngineer()
        
        numeric_columns = ['traffic_volume', 'average_speed', 'occupancy_rate', 'speed_ratio']
        df_processed = feature_engineer.create_lag_features(
            df_processed, numeric_columns, lags=[1, 2, 3, 6, 12, 24]
        )
        
        df_processed = feature_engineer.create_rolling_features(
            df_processed, numeric_columns, windows=[3, 6, 12, 24]
        )
        
        df_processed = feature_engineer.create_traffic_density_features(df_processed)
        
        df_processed = feature_engineer.create_spatial_features(df_processed)
        
        df_processed = df_processed.dropna()
        
        return df_processed
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        
        if time_split:
            df_sorted = df.sort_values('timestamp')
            n = len(df_sorted)
            
            train_end = int(n * (1 - test_size - validation_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df_sorted.iloc[:train_end]
            val_df = df_sorted.iloc[train_end:val_end]
            test_df = df_sorted.iloc[val_end:]
        else:
            train_df, temp_df = train_test_split(
                df, test_size=(test_size + validation_size), random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=(test_size / (test_size + validation_size)), random_state=42
            )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def scale_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale features and targets."""
        
        exclude_columns = [
            'timestamp', 'road_id', 'spatial_cluster', 'congestion_level',
            'latitude', 'longitude', 'lat_grid', 'lon_grid'
        ]
        
        self.feature_columns = [
            col for col in train_df.columns
            if col not in exclude_columns and train_df[col].dtype in ['int64', 'float64']
        ]
        
        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        
        y_train = train_df[self.target_column].values
        y_val = val_df[self.target_column].values
        y_test = test_df[self.target_column].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    def prepare_sequence_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM/RNN models."""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def save_preprocessed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """Save preprocessed data to file."""
        df.to_csv(file_path, index=False)
        logger.info(f"Saved preprocessed data to {file_path}")
    
    def load_preprocessed_data(self, file_path: str) -> pd.DataFrame:
        """Load preprocessed data from file."""
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        logger.info(f"Loaded preprocessed data from {file_path}")
        return df


def create_sample_data_pipeline() -> Dict[str, pd.DataFrame]:
    """Create sample data for testing the pipeline."""
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    locations = [
        (40.7589, -73.9851),  # Times Square
        (40.7505, -73.9934),  # Herald Square
        (40.7614, -73.9776),  # Central Park South
        (40.7831, -73.9712),  # Upper West Side
        (40.7282, -73.7949),  # Queens
    ]
    
    traffic_data = []
    for i, date in enumerate(dates[:1000]):  # Limit for demo
        for j, (lat, lon) in enumerate(locations):
            hour = date.hour
            day_of_week = date.weekday()
            
            if (7 <= hour <= 9) or (17 <= hour <= 19):
                base_volume = np.random.randint(800, 1200)
                base_speed = np.random.uniform(20, 40)
            elif day_of_week >= 5:  # Weekend
                base_volume = np.random.randint(400, 800)
                base_speed = np.random.uniform(40, 60)
            else:
                base_volume = np.random.randint(500, 900)
                base_speed = np.random.uniform(35, 55)
            
            traffic_data.append({
                'timestamp': date,
                'latitude': lat + np.random.uniform(-0.01, 0.01),
                'longitude': lon + np.random.uniform(-0.01, 0.01),
                'road_id': f'road_{j}',
                'road_type': np.random.choice(['highway', 'arterial', 'collector']),
                'traffic_volume': base_volume,
                'average_speed': base_speed,
                'free_flow_speed': 60.0,
                'occupancy_rate': np.random.uniform(20, 80),
                'address': f'Address {j}',
                'city': 'New York',
                'state': 'NY'
            })
    
    weather_data = []
    for date in dates[:100]:  # Limited sample
        weather_data.append({
            'timestamp': date,
            'latitude': 40.7589,
            'longitude': -73.9851,
            'temperature': np.random.uniform(-5, 35),
            'humidity': np.random.uniform(30, 90),
            'precipitation': np.random.exponential(0.5),
            'wind_speed': np.random.uniform(0, 30),
            'visibility': np.random.uniform(5, 20),
            'condition': np.random.choice(['clear', 'cloudy', 'rain', 'snow'])
        })
    
    event_data = []
    for i in range(50):  # 50 sample events
        event_data.append({
            'event_id': f'event_{i}',
            'timestamp': np.random.choice(dates[:100]),
            'latitude': np.random.choice([loc[0] for loc in locations]),
            'longitude': np.random.choice([loc[1] for loc in locations]),
            'event_type': np.random.choice(['concert', 'sports', 'festival', 'construction']),
            'expected_attendance': np.random.randint(1000, 50000),
            'duration_hours': np.random.uniform(1, 8),
            'impact_radius_km': np.random.uniform(2, 10)
        })
    
    return {
        'traffic': pd.DataFrame(traffic_data),
        'weather': pd.DataFrame(weather_data),
        'events': pd.DataFrame(event_data)
    }