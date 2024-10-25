"""
Synthetic traffic data generator for testing and demonstration.
Creates realistic traffic patterns with temporal, spatial, and weather dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from ..data.schema import (
    TrafficData, WeatherData, EventData, Location,
    RoadType, WeatherCondition, EventType, CongestionLevel,
    TimeFeatures, create_feature_dataframe
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDataGenerator:
    """Generates synthetic traffic data with realistic patterns."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the traffic data generator."""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        self.city_areas = {
            'city_center': {
                'center': (51.1694, 71.4491),  # Astana city center
                'radius_km': 3.0,
                'base_traffic_multiplier': 1.8,
                'rush_hour_multiplier': 2.5,
                'weekend_reduction': 0.6
            },
            'left_bank': {
                'center': (51.1280, 71.4030),  # Left bank (old city)
                'radius_km': 4.0,
                'base_traffic_multiplier': 1.5,
                'rush_hour_multiplier': 2.2,
                'weekend_reduction': 0.7
            },
            'right_bank': {
                'center': (51.1805, 71.4460),  # Right bank (new city)
                'radius_km': 5.0,
                'base_traffic_multiplier': 1.0,
                'rush_hour_multiplier': 1.8,
                'weekend_reduction': 0.8
            },
            'esil_district': {
                'center': (51.1350, 71.4700),  # Esil district
                'radius_km': 6.0,
                'base_traffic_multiplier': 0.8,
                'rush_hour_multiplier': 1.5,
                'weekend_reduction': 0.9
            }
        }
        
        self.road_characteristics = {
            RoadType.HIGHWAY: {
                'base_capacity': 4000,
                'speed_limit': 80,
                'weather_sensitivity': 0.3
            },
            RoadType.ARTERIAL: {
                'base_capacity': 2500,
                'speed_limit': 60,
                'weather_sensitivity': 0.4
            },
            RoadType.COLLECTOR: {
                'base_capacity': 1500,
                'speed_limit': 45,
                'weather_sensitivity': 0.5
            },
            RoadType.LOCAL: {
                'base_capacity': 800,
                'speed_limit': 35,
                'weather_sensitivity': 0.6
            },
            RoadType.RESIDENTIAL: {
                'base_capacity': 400,
                'speed_limit': 25,
                'weather_sensitivity': 0.7
            }
        }
    
    def generate_road_network(self, num_roads: int = 50) -> List[Dict]:
        """Generate a synthetic road network."""
        roads = []
        
        for i in range(num_roads):
            area_name = np.random.choice(list(self.city_areas.keys()))
            area = self.city_areas[area_name]
            
            center_lat, center_lon = area['center']
            radius_deg = area['radius_km'] / 111  # Approximate km to degrees
            
            start_lat = center_lat + np.random.uniform(-radius_deg, radius_deg)
            start_lon = center_lon + np.random.uniform(-radius_deg, radius_deg)
            
            segment_length_deg = np.random.uniform(0.002, 0.01)  # 0.2-1km segments
            end_lat = start_lat + np.random.uniform(-segment_length_deg, segment_length_deg)
            end_lon = start_lon + np.random.uniform(-segment_length_deg, segment_length_deg)
            
            if area_name == 'downtown':
                road_type = np.random.choice(
                    [RoadType.ARTERIAL, RoadType.COLLECTOR, RoadType.LOCAL],
                    p=[0.4, 0.4, 0.2]
                )
            elif area_name == 'suburban':
                road_type = np.random.choice(
                    [RoadType.HIGHWAY, RoadType.ARTERIAL, RoadType.COLLECTOR],
                    p=[0.3, 0.4, 0.3]
                )
            else:
                road_type = np.random.choice(
                    [RoadType.ARTERIAL, RoadType.COLLECTOR, RoadType.LOCAL],
                    p=[0.3, 0.5, 0.2]
                )
            
            segment_length = np.sqrt(
                (end_lat - start_lat) ** 2 + (end_lon - start_lon) ** 2
            ) * 111  # Convert to km
            
            road = {
                'road_id': f'road_{i:04d}',
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'length_km': segment_length,
                'road_type': road_type.value,
                'area': area_name,
                'speed_limit': self.road_characteristics[road_type]['speed_limit'],
                'capacity': self.road_characteristics[road_type]['base_capacity']
            }
            
            roads.append(road)
        
        logger.info(f"Generated {num_roads} roads")
        return roads
    
    def generate_traffic_data(
        self,
        roads: List[Dict],
        start_date: datetime,
        end_date: datetime,
        freq_minutes: int = 60
    ) -> List[TrafficData]:
        """Generate traffic data for roads over a time period."""
        traffic_data = []
        
        time_range = pd.date_range(start=start_date, end=end_date, freq=f'{freq_minutes}min')
        
        logger.info(f"Generating traffic data for {len(roads)} roads over {len(time_range)} time steps")
        
        for timestamp in time_range:
            for road_dict in roads:
                traffic_point = self._generate_traffic_point(road_dict, timestamp)
                if traffic_point:
                    traffic_data.append(traffic_point)
        
        logger.info(f"Generated {len(traffic_data)} traffic data points")
        return traffic_data
    
    def _generate_traffic_point(self, road_dict: Dict, timestamp: datetime) -> Optional[TrafficData]:
        """Generate a single traffic data point."""
        try:
            time_features = TimeFeatures.from_datetime(timestamp)
            
            road_type = RoadType(road_dict['road_type'])
            area = self.city_areas[road_dict['area']]
            road_chars = self.road_characteristics[road_type]
            
            base_volume = road_chars['base_capacity'] * np.random.uniform(0.3, 0.8)
            
            volume_multiplier = 1.0
            
            if time_features.is_rush_hour:
                volume_multiplier *= area['rush_hour_multiplier']
            
            if time_features.is_weekend:
                volume_multiplier *= area['weekend_reduction']
            
            hour_factors = {
                0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.15, 5: 0.3,
                6: 0.6, 7: 0.9, 8: 1.2, 9: 1.0, 10: 0.8, 11: 0.9,
                12: 1.0, 13: 0.9, 14: 0.8, 15: 0.9, 16: 1.1, 17: 1.3,
                18: 1.2, 19: 0.9, 20: 0.7, 21: 0.6, 22: 0.4, 23: 0.3
            }
            volume_multiplier *= hour_factors.get(time_features.hour, 1.0)
            
            volume_multiplier *= area['base_traffic_multiplier']
            
            volume_multiplier *= np.random.uniform(0.8, 1.2)
            
            traffic_volume = int(base_volume * volume_multiplier)
            traffic_volume = max(0, min(traffic_volume, road_chars['base_capacity']))
            
            occupancy_rate = (traffic_volume / road_chars['base_capacity']) * 100
            occupancy_rate += np.random.uniform(-10, 10)  # Add noise
            occupancy_rate = max(0, min(100, occupancy_rate))
            
            free_flow_speed = road_chars['speed_limit']
            congestion_factor = min(1.0, traffic_volume / road_chars['base_capacity'])
            
            speed_reduction = 1 / (1 + 0.15 * (congestion_factor ** 4))
            average_speed = free_flow_speed * speed_reduction
            
            average_speed *= np.random.uniform(0.9, 1.1)
            average_speed = max(5, min(average_speed, free_flow_speed))
            
            start_location = Location(
                latitude=road_dict['start_lat'],
                longitude=road_dict['start_lon']
            )
            
            traffic_data = TrafficData(
                timestamp=timestamp,
                location=start_location,
                road_id=road_dict['road_id'],
                road_type=road_type,
                traffic_volume=traffic_volume,
                average_speed=average_speed,
                free_flow_speed=free_flow_speed,
                occupancy_rate=occupancy_rate
            )
            
            return traffic_data
            
        except Exception as e:
            logger.warning(f"Error generating traffic point: {e}")
            return None
    
    def generate_weather_data(
        self,
        start_date: datetime,
        end_date: datetime,
        freq_hours: int = 1,
        num_stations: int = 5
    ) -> List[WeatherData]:
        """Generate weather data."""
        weather_data = []
        
        stations = []
        for i, (area_name, area) in enumerate(self.city_areas.items()):
            if i < num_stations:
                lat, lon = area['center']
                stations.append({
                    'station_id': f'weather_{i}',
                    'location': Location(lat, lon),
                    'area': area_name
                })
        
        time_range = pd.date_range(start=start_date, end=end_date, freq=f'{freq_hours}H')
        
        logger.info(f"Generating weather data for {len(stations)} stations over {len(time_range)} time steps")
        
        for timestamp in time_range:
            for station in stations:
                weather_point = self._generate_weather_point(station, timestamp)
                if weather_point:
                    weather_data.append(weather_point)
        
        logger.info(f"Generated {len(weather_data)} weather data points")
        return weather_data
    
    def _generate_weather_point(self, station: Dict, timestamp: datetime) -> Optional[WeatherData]:
        """Generate a single weather data point."""
        try:
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            hour_temp_variation = 5 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)
            
            temp_noise = np.random.normal(0, 3)
            
            temperature = seasonal_temp + hour_temp_variation + temp_noise
            
            base_humidity = 70 - (temperature - 15) * 1.5
            humidity = base_humidity + np.random.normal(0, 10)
            humidity = max(20, min(95, humidity))
            
            precipitation = 0.0
            if np.random.random() < 0.1:  # 10% chance of precipitation
                precipitation = np.random.exponential(2.0)
            
            wind_speed = np.random.gamma(2, 3)  # Gamma distribution for wind
            
            base_visibility = 20
            if precipitation > 0:
                visibility = base_visibility * (1 - min(0.8, precipitation / 10))
            else:
                visibility = base_visibility + np.random.uniform(-2, 2)
            visibility = max(1, visibility)
            
            if precipitation > 5:
                condition = WeatherCondition.RAIN
            elif precipitation > 0:
                condition = WeatherCondition.CLOUDY
            elif visibility < 5:
                condition = WeatherCondition.FOG
            elif temperature < 0 and precipitation > 0:
                condition = WeatherCondition.SNOW
            else:
                condition = np.random.choice([
                    WeatherCondition.CLEAR,
                    WeatherCondition.CLOUDY
                ], p=[0.7, 0.3])
            
            weather_data = WeatherData(
                timestamp=timestamp,
                location=station['location'],
                temperature=temperature,
                humidity=humidity,
                precipitation=precipitation,
                wind_speed=wind_speed,
                visibility=visibility,
                condition=condition
            )
            
            return weather_data
            
        except Exception as e:
            logger.warning(f"Error generating weather point: {e}")
            return None
    
    def generate_event_data(
        self,
        start_date: datetime,
        end_date: datetime,
        avg_events_per_day: float = 2.0
    ) -> List[EventData]:
        """Generate event data."""
        event_data = []
        
        total_days = (end_date - start_date).days
        total_events = int(total_days * avg_events_per_day)
        
        logger.info(f"Generating {total_events} events over {total_days} days")
        
        for i in range(total_events):
            event = self._generate_event(start_date, end_date, i)
            if event:
                event_data.append(event)
        
        logger.info(f"Generated {len(event_data)} events")
        return event_data
    
    def _generate_event(self, start_date: datetime, end_date: datetime, event_id: int) -> Optional[EventData]:
        """Generate a single event."""
        try:
            total_seconds = int((end_date - start_date).total_seconds())
            random_seconds = np.random.randint(0, total_seconds)
            timestamp = start_date + timedelta(seconds=random_seconds)
            
            area_probs = [0.4, 0.3, 0.2, 0.1]  # downtown, midtown, residential, suburban
            area_name = np.random.choice(list(self.city_areas.keys()), p=area_probs)
            area = self.city_areas[area_name]
            
            center_lat, center_lon = area['center']
            radius_deg = area['radius_km'] / 111
            
            event_lat = center_lat + np.random.uniform(-radius_deg, radius_deg)
            event_lon = center_lon + np.random.uniform(-radius_deg, radius_deg)
            
            location = Location(latitude=event_lat, longitude=event_lon)
            
            if 7 <= timestamp.hour <= 19:  # Daytime
                event_type = np.random.choice([
                    EventType.CONSTRUCTION,
                    EventType.FESTIVAL,
                    EventType.SPORTS,
                    EventType.CONCERT
                ], p=[0.4, 0.2, 0.2, 0.2])
            else:  # Nighttime
                event_type = np.random.choice([
                    EventType.ACCIDENT,
                    EventType.CONCERT,
                    EventType.CONSTRUCTION
                ], p=[0.5, 0.3, 0.2])
            
            if event_type == EventType.CONSTRUCTION:
                expected_attendance = None
                duration_hours = np.random.uniform(4, 12)
                impact_radius = np.random.uniform(0.5, 2.0)
            elif event_type in [EventType.SPORTS, EventType.CONCERT]:
                expected_attendance = np.random.randint(1000, 50000)
                duration_hours = np.random.uniform(2, 6)
                impact_radius = np.random.uniform(2, 8)
            elif event_type == EventType.FESTIVAL:
                expected_attendance = np.random.randint(5000, 100000)
                duration_hours = np.random.uniform(6, 16)
                impact_radius = np.random.uniform(3, 10)
            else:  # ACCIDENT
                expected_attendance = None
                duration_hours = np.random.uniform(0.5, 4)
                impact_radius = np.random.uniform(0.2, 1.5)
            
            event_data = EventData(
                event_id=f'event_{event_id:04d}',
                timestamp=timestamp,
                location=location,
                event_type=event_type,
                expected_attendance=expected_attendance,
                duration_hours=duration_hours,
                impact_radius_km=impact_radius
            )
            
            return event_data
            
        except Exception as e:
            logger.warning(f"Error generating event: {e}")
            return None
    
    def generate_complete_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        num_roads: int = 50,
        traffic_freq_minutes: int = 60,
        weather_freq_hours: int = 1,
        events_per_day: float = 2.0,
        output_dir: str = "data/generated"
    ) -> Dict[str, pd.DataFrame]:
        """Generate a complete synthetic dataset."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating complete synthetic dataset...")
        
        roads = self.generate_road_network(num_roads)
        roads_df = pd.DataFrame(roads)
        
        traffic_data = self.generate_traffic_data(
            roads, start_date, end_date, traffic_freq_minutes
        )
        
        traffic_records = []
        for traffic in traffic_data:
            road = next((r for r in roads if r['road_id'] == traffic.road_id), None)
            area = road['area'] if road else 'unknown'
            
            district_mapping = {
                'city_center': 'City Center',
                'left_bank': 'Left Bank', 
                'right_bank': 'Right Bank',
                'esil_district': 'Esil District',
                'suburban': 'Suburban',
                'downtown': 'City Center',  # fallback
                'unknown': 'Suburban'  # fallback
            }
            district = district_mapping.get(area, 'Suburban')
            
            record = {
                'timestamp': traffic.timestamp,
                'latitude': traffic.location.latitude,
                'longitude': traffic.location.longitude,
                'road_id': traffic.road_id,
                'road_type': traffic.road_type.value,
                'traffic_volume': traffic.traffic_volume,
                'average_speed': traffic.average_speed,
                'free_flow_speed': traffic.free_flow_speed,
                'occupancy_rate': traffic.occupancy_rate,
                'congestion_level': traffic.congestion_level.value,
                'congestion_name': traffic.congestion_level.name,
                'district': district
            }
            traffic_records.append(record)
        
        traffic_df = pd.DataFrame(traffic_records)
        
        weather_data = self.generate_weather_data(
            start_date, end_date, weather_freq_hours
        )
        
        weather_records = []
        for weather in weather_data:
            record = {
                'timestamp': weather.timestamp,
                'latitude': weather.location.latitude,
                'longitude': weather.location.longitude,
                'temperature': weather.temperature,
                'humidity': weather.humidity,
                'precipitation': weather.precipitation,
                'wind_speed': weather.wind_speed,
                'visibility': weather.visibility,
                'condition': weather.condition.value
            }
            weather_records.append(record)
        
        weather_df = pd.DataFrame(weather_records)
        
        event_data = self.generate_event_data(start_date, end_date, events_per_day)
        
        event_records = []
        for event in event_data:
            record = {
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'latitude': event.location.latitude,
                'longitude': event.location.longitude,
                'event_type': event.event_type.value,
                'expected_attendance': event.expected_attendance,
                'duration_hours': event.duration_hours,
                'impact_radius_km': event.impact_radius_km
            }
            event_records.append(record)
        
        events_df = pd.DataFrame(event_records)
        
        logger.info("Creating feature dataset...")
        feature_df = create_feature_dataframe(traffic_data, weather_data, event_data)
        
        datasets = {
            'roads': roads_df,
            'traffic': traffic_df,
            'weather': weather_df,
            'events': events_df,
            'features': feature_df
        }
        
        for name, df in datasets.items():
            file_path = output_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {name} dataset to {file_path} ({len(df)} records)")
        
        logger.info("Dataset generation completed!")
        return datasets


def create_demo_dataset() -> Dict[str, pd.DataFrame]:
    """Create a small demo dataset for testing."""
    generator = TrafficDataGenerator(random_seed=42)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)  # One week
    
    return generator.generate_complete_dataset(
        start_date=start_date,
        end_date=end_date,
        num_roads=20,
        traffic_freq_minutes=60,
        weather_freq_hours=2,
        events_per_day=1.0,
        output_dir="data/demo"
    )


if __name__ == "__main__":
    demo_data = create_demo_dataset()
    print("Demo dataset created successfully!")
    
    for name, df in demo_data.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if not df.empty:
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Sample row:\n{df.iloc[0].to_dict()}")