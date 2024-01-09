#!/usr/bin/env python3
"""
Sample data generation script for the Traffic Congestion Prediction System.
This script generates synthetic traffic data for testing and demonstration purposes.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.generator import TrafficDataGenerator
from src.data.schema import Location
from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


def get_district_from_road(road_id: str, roads: list) -> str:
    """Map road ID to district name."""
    # Find the road to get area information
    road = next((r for r in roads if r['road_id'] == road_id), None)
    area = road['area'] if road else 'unknown'
    
    # Map area to district name
    district_mapping = {
        'city_center': 'City Center',
        'left_bank': 'Left Bank', 
        'right_bank': 'Right Bank',
        'esil_district': 'Esil District',
        'suburban': 'Suburban',
        'downtown': 'City Center',  # fallback
        'unknown': 'Suburban'  # fallback
    }
    return district_mapping.get(area, 'Suburban')


def generate_sample_datasets():
    """Generate various sample datasets for different use cases."""
    
    logger.info("Starting sample data generation...")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "samples").mkdir(exist_ok=True)
    
    # Initialize data generator
    generator = TrafficDataGenerator()
    
    # Generate different datasets
    datasets = {
        "astana_week": {
            "description": "One week of traffic data for Astana city center",
            "center_location": Location(latitude=51.1694, longitude=71.4491),  # Astana center
            "radius_km": 5.0,
            "num_roads": 30,
            "start_date": datetime.now() - timedelta(days=7),
            "end_date": datetime.now(),
            "frequency_minutes": 15
        },
        "left_bank_day": {
            "description": "One day of high-resolution traffic data for Left Bank",
            "center_location": Location(latitude=51.1280, longitude=71.4030),  # Left bank
            "radius_km": 3.0,
            "num_roads": 20,
            "start_date": datetime.now() - timedelta(days=1),
            "end_date": datetime.now(),
            "frequency_minutes": 5
        },
        "right_bank_month": {
            "description": "One month of traffic data for Right Bank (new city)",
            "center_location": Location(latitude=51.1805, longitude=71.4460),  # Right bank
            "radius_km": 8.0,
            "num_roads": 50,
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "frequency_minutes": 30
        },
        "highways_week": {
            "description": "Highway traffic data around Astana with heavy congestion patterns",
            "center_location": Location(latitude=51.1694, longitude=71.4491),  # Astana
            "radius_km": 15.0,
            "num_roads": 25,
            "start_date": datetime.now() - timedelta(days=7),
            "end_date": datetime.now(),
            "frequency_minutes": 10,
            "highway_focus": True
        },
        "small_test": {
            "description": "Small dataset for quick testing in Astana",
            "center_location": Location(latitude=51.1694, longitude=71.4491),
            "radius_km": 1.0,
            "num_roads": 5,
            "start_date": datetime.now() - timedelta(hours=24),
            "end_date": datetime.now(),
            "frequency_minutes": 60
        }
    }
    
    generated_files = {}
    
    for dataset_name, params in datasets.items():
        logger.info(f"Generating dataset: {dataset_name}")
        
        try:
            # Generate road network first
            roads = generator.generate_road_network(num_roads=params["num_roads"])
            
            # Generate traffic data
            traffic_data = generator.generate_traffic_data(
                roads=roads,
                start_date=params["start_date"],
                end_date=params["end_date"],
                freq_minutes=params["frequency_minutes"]
            )
            
            # Generate weather data
            weather_data = generator.generate_weather_data(
                start_date=params["start_date"],
                end_date=params["end_date"],
                freq_hours=1,
                num_stations=5
            )
            
            # Generate event data
            event_data = generator.generate_event_data(
                start_date=params["start_date"],
                end_date=params["end_date"],
                avg_events_per_day=2.0
            )
            
            # Convert to DataFrames
            traffic_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'road_id': t.road_id,
                    'latitude': t.location.latitude,
                    'longitude': t.location.longitude,
                    'traffic_volume': t.traffic_volume,
                    'average_speed': t.average_speed,
                    'free_flow_speed': t.free_flow_speed,
                    'occupancy_rate': t.occupancy_rate,
                    'congestion_level': t.congestion_level.value,
                    'congestion_name': t.congestion_level.name,
                    'road_type': t.road_type.value,
                    'district': get_district_from_road(t.road_id, roads)
                }
                for t in traffic_data
            ])
            
            weather_df = pd.DataFrame([
                {
                    'timestamp': w.timestamp,
                    'latitude': w.location.latitude,
                    'longitude': w.location.longitude,
                    'temperature': w.temperature,
                    'humidity': w.humidity,
                    'precipitation': w.precipitation,
                    'wind_speed': w.wind_speed,
                    'visibility': w.visibility,
                    'weather_condition': w.condition.value
                }
                for w in weather_data
            ])
            
            event_df = pd.DataFrame([
                {
                    'timestamp': e.timestamp,
                    'event_id': e.event_id,
                    'latitude': e.location.latitude,
                    'longitude': e.location.longitude,
                    'event_type': e.event_type.value,
                    'expected_attendance': e.expected_attendance,
                    'duration_hours': e.duration_hours,
                    'impact_radius_km': e.impact_radius_km
                }
                for e in event_data
            ])
            
            # Save datasets
            base_path = data_dir / "samples" / dataset_name
            base_path.mkdir(exist_ok=True)
            
            # Save as CSV
            traffic_file = base_path / "traffic_data.csv"
            weather_file = base_path / "weather_data.csv"
            events_file = base_path / "events_data.csv"
            
            traffic_df.to_csv(traffic_file, index=False)
            weather_df.to_csv(weather_file, index=False)
            event_df.to_csv(events_file, index=False)
            
            # Save as Parquet for better performance
            traffic_df.to_parquet(base_path / "traffic_data.parquet", index=False)
            weather_df.to_parquet(base_path / "weather_data.parquet", index=False)
            event_df.to_parquet(base_path / "events_data.parquet", index=False)
            
            # Create metadata file
            metadata = {
                "dataset_name": dataset_name,
                "description": params["description"],
                "generated_at": datetime.now().isoformat(),
                "parameters": {
                    "center_lat": params["center_location"].latitude,
                    "center_lon": params["center_location"].longitude,
                    "radius_km": params["radius_km"],
                    "num_roads": params["num_roads"],
                    "start_date": params["start_date"].isoformat(),
                    "end_date": params["end_date"].isoformat(),
                    "frequency_minutes": params["frequency_minutes"]
                },
                "statistics": {
                    "traffic_records": len(traffic_df),
                    "weather_records": len(weather_df),
                    "event_records": len(event_df),
                    "date_range_days": (params["end_date"] - params["start_date"]).days,
                    "unique_roads": traffic_df['road_id'].nunique(),
                    "congestion_distribution": traffic_df['congestion_name'].value_counts().to_dict()
                },
                "files": {
                    "traffic_csv": str(traffic_file.name),
                    "weather_csv": str(weather_file.name),
                    "events_csv": str(events_file.name),
                    "traffic_parquet": "traffic_data.parquet",
                    "weather_parquet": "weather_data.parquet",
                    "events_parquet": "events_data.parquet"
                }
            }
            
            with open(base_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            generated_files[dataset_name] = {
                "path": str(base_path),
                "traffic_records": len(traffic_df),
                "weather_records": len(weather_df),
                "event_records": len(event_df)
            }
            
            logger.info(f"Generated {dataset_name}: {len(traffic_df)} traffic records, "
                       f"{len(weather_df)} weather records, {len(event_df)} events")
            
        except Exception as e:
            logger.error(f"Failed to generate dataset {dataset_name}: {str(e)}")
            continue
    
    # Create summary file
    summary = {
        "generation_completed_at": datetime.now().isoformat(),
        "total_datasets": len(generated_files),
        "datasets": generated_files,
        "total_records": {
            "traffic": sum(d["traffic_records"] for d in generated_files.values()),
            "weather": sum(d["weather_records"] for d in generated_files.values()),
            "events": sum(d["event_records"] for d in generated_files.values())
        }
    }
    
    with open(data_dir / "samples" / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Sample data generation completed. Generated {len(generated_files)} datasets.")
    logger.info(f"Total records: {summary['total_records']}")
    
    return generated_files


def create_demo_scenarios():
    """Create specific scenarios for demonstrations."""
    
    logger.info("Creating demo scenarios...")
    
    data_dir = Path("data") / "demos"
    data_dir.mkdir(exist_ok=True)
    
    generator = TrafficDataGenerator()
    
    # Scenario 1: Rush hour congestion
    logger.info("Creating rush hour congestion scenario...")
    
    # Morning rush hour (8-10 AM) with heavy congestion (Astana timezone)
    rush_start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    rush_end = rush_start + timedelta(hours=2)
    
    # Generate roads for rush hour scenario in Astana
    rush_roads = generator.generate_road_network(num_roads=15)
    
    rush_traffic = generator.generate_traffic_data(
        roads=rush_roads,
        start_date=rush_start,
        end_date=rush_end,
        freq_minutes=5
    )
    
    rush_df = pd.DataFrame([
        {
            'timestamp': t.timestamp,
            'road_id': t.road_id,
            'latitude': t.location.latitude,
            'longitude': t.location.longitude,
            'traffic_volume': t.traffic_volume,
            'average_speed': t.average_speed,
            'free_flow_speed': t.free_flow_speed,
            'occupancy_rate': t.occupancy_rate,
            'congestion_level': t.congestion_level.value,
            'congestion_name': t.congestion_level.name,
            'road_type': t.road_type.value,
            'district': get_district_from_road(t.road_id, rush_roads)
        }
        for t in rush_traffic
    ])
    
    rush_df.to_csv(data_dir / "astana_rush_hour_congestion.csv", index=False)
    
    # Scenario 2: Weather impact in Astana
    logger.info("Creating weather impact scenario for Astana...")
    
    # Traffic during a snowy day (common in Astana winters)
    weather_start = datetime.now() - timedelta(hours=12)
    weather_end = datetime.now()
    
    # Generate roads for weather scenario in Astana
    weather_roads = generator.generate_road_network(num_roads=20)
    
    weather_traffic = generator.generate_traffic_data(
        roads=weather_roads,
        start_date=weather_start,
        end_date=weather_end,
        freq_minutes=15
    )
    
    # Generate corresponding weather data with rain
    rainy_weather = generator.generate_weather_data(
        start_date=weather_start,
        end_date=weather_end,
        freq_hours=1,
        num_stations=3
    )
    
    weather_traffic_df = pd.DataFrame([
        {
            'timestamp': t.timestamp,
            'road_id': t.road_id,
            'latitude': t.location.latitude,
            'longitude': t.location.longitude,
            'traffic_volume': t.traffic_volume,
            'average_speed': t.average_speed,
            'free_flow_speed': t.free_flow_speed,
            'occupancy_rate': t.occupancy_rate,
            'congestion_level': t.congestion_level.value,
            'congestion_name': t.congestion_level.name,
            'road_type': t.road_type.value,
            'district': get_district_from_road(t.road_id, weather_roads)
        }
        for t in weather_traffic
    ])
    
    weather_df = pd.DataFrame([
        {
            'timestamp': w.timestamp,
            'temperature': w.temperature,
            'humidity': w.humidity,
            'precipitation': w.precipitation,
            'wind_speed': w.wind_speed,
            'visibility': w.visibility,
            'weather_condition': w.condition.value
        }
        for w in rainy_weather
    ])
    
    weather_traffic_df.to_csv(data_dir / "astana_winter_traffic.csv", index=False)
    weather_df.to_csv(data_dir / "astana_winter_weather.csv", index=False)
    
    logger.info("Demo scenarios created successfully!")


if __name__ == "__main__":
    # Generate sample datasets
    generated_files = generate_sample_datasets()
    
    # Create demo scenarios
    create_demo_scenarios()
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATION COMPLETED")
    print("="*60)
    print(f"Generated {len(generated_files)} datasets:")
    
    for name, info in generated_files.items():
        print(f"  • {name}: {info['traffic_records']} traffic records")
    
    print("\nDatasets saved in: data/samples/")
    print("Demo scenarios saved in: data/demos/")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")
    print("="*60)