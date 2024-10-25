"""
Data schema and models for traffic congestion prediction system.
Defines the structure for traffic data, weather data, and event data.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


class RoadType(Enum):
    """Enumeration for different types of roads."""
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    COLLECTOR = "collector"
    LOCAL = "local"
    RESIDENTIAL = "residential"


class WeatherCondition(Enum):
    """Enumeration for weather conditions."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"


class EventType(Enum):
    """Enumeration for event types that affect traffic."""
    CONCERT = "concert"
    SPORTS = "sports"
    FESTIVAL = "festival"
    CONSTRUCTION = "construction"
    ACCIDENT = "accident"
    HOLIDAY = "holiday"
    RUSH_HOUR = "rush_hour"


class CongestionLevel(Enum):
    """Enumeration for traffic congestion levels."""
    FREE_FLOW = 0  # Speed >= 80% of free flow speed
    LIGHT = 1      # Speed 60-80% of free flow speed
    MODERATE = 2   # Speed 40-60% of free flow speed
    HEAVY = 3      # Speed 20-40% of free flow speed
    SEVERE = 4     # Speed < 20% of free flow speed


@dataclass
class Location:
    """Represents a geographical location."""
    latitude: float
    longitude: float
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    def __post_init__(self):
        """Validate coordinates."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= self.longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")


@dataclass
class TrafficData:
    """Schema for traffic data points."""
    timestamp: datetime
    location: Location
    road_id: str
    road_type: RoadType
    traffic_volume: int  # vehicles per hour
    average_speed: float  # km/h
    free_flow_speed: float  # km/h (speed limit or typical speed)
    occupancy_rate: float  # percentage (0-100)
    congestion_level: Optional[CongestionLevel] = None
    
    def __post_init__(self):
        """Calculate congestion level based on speed ratio."""
        if self.congestion_level is None:
            speed_ratio = self.average_speed / self.free_flow_speed
            if speed_ratio >= 0.8:
                self.congestion_level = CongestionLevel.FREE_FLOW
            elif speed_ratio >= 0.6:
                self.congestion_level = CongestionLevel.LIGHT
            elif speed_ratio >= 0.4:
                self.congestion_level = CongestionLevel.MODERATE
            elif speed_ratio >= 0.2:
                self.congestion_level = CongestionLevel.HEAVY
            else:
                self.congestion_level = CongestionLevel.SEVERE


@dataclass
class WeatherData:
    """Schema for weather data."""
    timestamp: datetime
    location: Location
    temperature: float  # Celsius
    humidity: float  # percentage (0-100)
    precipitation: float  # mm/hour
    wind_speed: float  # km/h
    visibility: float  # kilometers
    condition: WeatherCondition
    
    def to_features(self) -> Dict[str, float]:
        """Convert weather data to numerical features."""
        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'precipitation': self.precipitation,
            'wind_speed': self.wind_speed,
            'visibility': self.visibility,
            'weather_clear': 1.0 if self.condition == WeatherCondition.CLEAR else 0.0,
            'weather_rain': 1.0 if self.condition == WeatherCondition.RAIN else 0.0,
            'weather_snow': 1.0 if self.condition == WeatherCondition.SNOW else 0.0,
            'weather_fog': 1.0 if self.condition == WeatherCondition.FOG else 0.0,
        }


@dataclass
class EventData:
    """Schema for events that affect traffic."""
    event_id: str
    timestamp: datetime
    location: Location
    event_type: EventType
    expected_attendance: Optional[int] = None
    duration_hours: Optional[float] = None
    impact_radius_km: float = 5.0  # Expected impact radius
    
    def to_features(self) -> Dict[str, float]:
        """Convert event data to numerical features."""
        return {
            'event_concert': 1.0 if self.event_type == EventType.CONCERT else 0.0,
            'event_sports': 1.0 if self.event_type == EventType.SPORTS else 0.0,
            'event_festival': 1.0 if self.event_type == EventType.FESTIVAL else 0.0,
            'event_construction': 1.0 if self.event_type == EventType.CONSTRUCTION else 0.0,
            'event_accident': 1.0 if self.event_type == EventType.ACCIDENT else 0.0,
            'expected_attendance': float(self.expected_attendance or 0),
            'duration_hours': float(self.duration_hours or 0),
            'impact_radius_km': self.impact_radius_km,
        }


@dataclass
class TimeFeatures:
    """Time-based features extracted from datetime."""
    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    day_of_month: int
    month: int
    is_weekend: bool
    is_rush_hour: bool
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'TimeFeatures':
        """Create TimeFeatures from datetime object."""
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5
        is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
        
        return cls(
            hour=hour,
            day_of_week=day_of_week,
            day_of_month=dt.day,
            month=dt.month,
            is_weekend=is_weekend,
            is_rush_hour=is_rush_hour
        )
    
    def to_features(self) -> Dict[str, float]:
        """Convert time features to numerical format."""
        return {
            'hour': float(self.hour),
            'day_of_week': float(self.day_of_week),
            'day_of_month': float(self.day_of_month),
            'month': float(self.month),
            'is_weekend': 1.0 if self.is_weekend else 0.0,
            'is_rush_hour': 1.0 if self.is_rush_hour else 0.0,
            'hour_sin': np.sin(2 * np.pi * self.hour / 24),
            'hour_cos': np.cos(2 * np.pi * self.hour / 24),
            'day_sin': np.sin(2 * np.pi * self.day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * self.day_of_week / 7),
            'month_sin': np.sin(2 * np.pi * self.month / 12),
            'month_cos': np.cos(2 * np.pi * self.month / 12),
        }


class DataValidator:
    """Validates data integrity and consistency."""
    
    @staticmethod
    def validate_traffic_data(data: TrafficData) -> bool:
        """Validate traffic data point."""
        if data.traffic_volume < 0:
            return False
        if data.average_speed < 0:
            return False
        if data.free_flow_speed <= 0:
            return False
        if not (0 <= data.occupancy_rate <= 100):
            return False
        return True
    
    @staticmethod
    def validate_weather_data(data: WeatherData) -> bool:
        """Validate weather data point."""
        if not (-50 <= data.temperature <= 60):  # Reasonable temperature range
            return False
        if not (0 <= data.humidity <= 100):
            return False
        if data.precipitation < 0:
            return False
        if data.wind_speed < 0:
            return False
        if data.visibility < 0:
            return False
        return True
    
    @staticmethod
    def validate_event_data(data: EventData) -> bool:
        """Validate event data point."""
        if data.expected_attendance is not None and data.expected_attendance < 0:
            return False
        if data.duration_hours is not None and data.duration_hours < 0:
            return False
        if data.impact_radius_km < 0:
            return False
        return True


def create_feature_dataframe(
    traffic_data: List[TrafficData],
    weather_data: List[WeatherData] = None,
    event_data: List[EventData] = None
) -> pd.DataFrame:
    """
    Create a feature dataframe from traffic, weather, and event data.
    
    Args:
        traffic_data: List of traffic data points
        weather_data: Optional list of weather data points
        event_data: Optional list of event data points
    
    Returns:
        DataFrame with features for machine learning
    """
    features = []
    
    for traffic in traffic_data:
        feature_row = {
            'timestamp': traffic.timestamp,
            'latitude': traffic.location.latitude,
            'longitude': traffic.location.longitude,
            'road_id': traffic.road_id,
            'traffic_volume': traffic.traffic_volume,
            'average_speed': traffic.average_speed,
            'free_flow_speed': traffic.free_flow_speed,
            'occupancy_rate': traffic.occupancy_rate,
            'congestion_level': traffic.congestion_level.value,
            'speed_ratio': traffic.average_speed / traffic.free_flow_speed,
        }
        
        for road_type in RoadType:
            feature_row[f'road_type_{road_type.value}'] = (
                1.0 if traffic.road_type == road_type else 0.0
            )
        
        time_features = TimeFeatures.from_datetime(traffic.timestamp)
        feature_row.update(time_features.to_features())
        
        if weather_data:
            closest_weather = min(
                weather_data,
                key=lambda w: abs((w.timestamp - traffic.timestamp).total_seconds())
            )
            feature_row.update(closest_weather.to_features())
        
        if event_data:
            active_events = [
                event for event in event_data
                if abs((event.timestamp - traffic.timestamp).total_seconds()) <= 3600  # Within 1 hour
            ]
            
            event_features = {
                'event_concert': 0.0,
                'event_sports': 0.0,
                'event_festival': 0.0,
                'event_construction': 0.0,
                'event_accident': 0.0,
                'total_expected_attendance': 0.0,
                'event_count': len(active_events),
            }
            
            for event in active_events:
                event_feature_dict = event.to_features()
                for key, value in event_feature_dict.items():
                    if key.startswith('event_'):
                        event_features[key] = max(event_features.get(key, 0.0), value)
                    elif key == 'expected_attendance':
                        event_features['total_expected_attendance'] += value
            
            feature_row.update(event_features)
        
        features.append(feature_row)
    
    return pd.DataFrame(features)