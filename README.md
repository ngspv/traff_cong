# AI Model for Predicting Traffic Congestion in Astana

A comprehensive traffic congestion prediction system for Astana, Kazakhstan using machine learning models (LSTM and Temporal Fusion Transformer) with interactive web interface, route optimization, and real-time visualization capabilities.

## Overview

This project implements an advanced traffic congestion prediction system specifically designed for Astana (Nur-Sultan), Kazakhstan that combines:

- **Machine Learning Models**: LSTM and Temporal Fusion Transformer (TFT) for time-series prediction
- **Route Optimization**: OSMnx-based routing with congestion-aware pathfinding
- **Interactive Web Interface**: Streamlit-based dashboard for data upload, prediction, and visualization
- **Real-time Visualization**: Folium maps with traffic heatmaps, route visualization, and analytics
- **Synthetic Data Generation**: Realistic traffic data simulation for testing and demonstration

## Features

### Core Functionality
- **Traffic Prediction**: Predict congestion levels using LSTM and TFT models
- **Route Optimization**: Find optimal routes considering real-time traffic conditions
- **Data Processing**: Comprehensive pipeline for traffic, weather, and event data
- **Visualization**: Interactive maps and charts for traffic analysis
- **Real-time Analytics**: Dashboard with congestion patterns and trends

### Models Implemented
1. **LSTM Model**: Sequential neural network with attention mechanism
2. **Temporal Fusion Transformer**: State-of-the-art interpretable forecasting model
3. **Route Optimizer**: Graph-based pathfinding with congestion weights

### Data Sources Supported
- Traffic volume and speed data
- Weather conditions (temperature, humidity, wind, visibility)
- Special events (accidents, construction, events)
- Road network topology
- Historical patterns

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Git

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd traff_cong
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Generate sample data** (optional):
```bash
python generate_sample_data.py
```

5. **Run the application**:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
traff_cong/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── generate_sample_data.py         # Sample data generation script
├── .env.example                    # Environment variables template
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py               # Data models and schemas
│   │   ├── pipeline.py             # Data processing pipeline
│   │   └── generator.py            # Synthetic data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM implementation
│   │   └── tft_model.py            # Temporal Fusion Transformer
│   └── utils/
│       ├── __init__.py
│       ├── routing.py              # Route optimization
│       ├── visualization.py        # Map and chart visualization
│       └── logger.py               # Logging configuration
├── data/                           # Data storage directory
│   ├── samples/                    # Sample datasets
│   ├── demos/                      # Demo scenarios
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed datasets
├── models/                         # Trained model storage
└── logs/                           # Application logs
```

## Usage Guide

### 1. Data Upload & Processing

**Via Web Interface**:
1. Open the Streamlit app
2. Navigate to "Data Upload" page
3. Upload CSV/Excel files with traffic data
4. Review data summary and proceed to processing

**Supported Data Format**:
```csv
timestamp,road_id,latitude,longitude,traffic_volume,average_speed,occupancy_rate,congestion_level,road_type
2024-01-01 08:00:00,R001,40.7589,-73.9851,1200,25.5,75.2,2,highway
```

### 2. Traffic Prediction

**Using LSTM Model**:
```python
from src.models.lstm_model import LSTMTrafficPredictor

# Initialize model
lstm_model = LSTMTrafficPredictor()

# Train on historical data
lstm_model.train(training_data, validation_data)

# Make predictions
predictions = lstm_model.predict(input_sequences)
```

**Using TFT Model**:
```python
from src.models.tft_model import TFTTrafficPredictor

# Initialize model
tft_model = TFTTrafficPredictor()

# Train model
tft_model.train(training_data)

# Generate forecasts
forecasts = tft_model.predict(prediction_data)
```

### 3. Route Optimization

```python
from src.utils.routing import RouteOptimizer

# Initialize optimizer
optimizer = RouteOptimizer()

# Find optimal routes
routes = optimizer.find_optimal_routes(
    start_location=(40.7589, -73.9851),
    end_location=(40.7505, -73.9934),
    current_traffic_data=traffic_data
)
```

### 4. Visualization

**Interactive Maps**:
```python
from src.utils.visualization import TrafficMapVisualizer

# Create visualizer
visualizer = TrafficMapVisualizer()

# Create base map
map_obj = visualizer.create_base_map()

# Add traffic layers
map_obj = visualizer.add_traffic_heatmap(map_obj, traffic_data)
map_obj = visualizer.add_traffic_markers(map_obj, traffic_data)

# Display routes
map_obj = visualizer.add_route_visualization(map_obj, routes)
```

## Web Interface Guide

### Navigation
The Streamlit interface consists of four main pages:

1. **Data Upload**: Upload and preview traffic datasets
2. **Prediction**: Train models and generate traffic forecasts  
3. **Route Planning**: Optimize routes based on traffic conditions
4. **Analytics**: View traffic patterns and performance metrics

### Key Features

**Data Upload Page**:
- File upload (CSV, Excel, JSON, Parquet)
- Data validation and preview
- Summary statistics
- Data quality assessment

**Prediction Page**:
- Model selection (LSTM vs TFT)
- Training progress monitoring
- Prediction visualization
- Model performance metrics

**Route Planning Page**:
- Interactive map for start/end selection
- Real-time route optimization
- Alternative route comparison
- Traffic-aware time estimates

**Analytics Page**:
- Traffic pattern analysis
- Congestion heatmaps
- Performance dashboards
- Historical trend analysis

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Environment
ENVIRONMENT=development

# API Keys (optional)
WEATHER_API_KEY=your_openweather_api_key
TRAFFIC_API_KEY=your_traffic_api_key

# Database (if using external DB)
DATABASE_HOST=localhost
DATABASE_NAME=traffic_db
```

### Model Configuration

Edit `config.py` to customize:

```python
# LSTM Model Settings
config.lstm_model.sequence_length = 24      # Hours of historical data
config.lstm_model.prediction_horizon = 6    # Hours to predict
config.lstm_model.lstm_units = [128, 64, 32] # Network architecture

# TFT Model Settings  
config.tft_model.max_encoder_length = 24    # Historical context
config.tft_model.max_prediction_length = 6  # Prediction horizon
config.tft_model.hidden_size = 32           # Model complexity

# Visualization Settings for Astana
config.visualization.default_map_center = (51.1694, 71.4491)  # Astana center
config.visualization.clustering_enabled = True
```

## Testing

### Generate Sample Data
```bash
python generate_sample_data.py
```

This creates:
- `data/samples/`: Various test datasets
- `data/demos/`: Specific demo scenarios

### Run System Tests
```bash
# Test individual components
python -m pytest tests/

# Test data pipeline
python -c "from src.data.pipeline import DataProcessor; dp = DataProcessor(); print('Pipeline OK')"

# Test models
python -c "from src.models.lstm_model import LSTMTrafficPredictor; print('LSTM Model OK')"

# Test visualization
python -c "from src.utils.visualization import TrafficMapVisualizer; print('Visualization OK')"
```

## API Reference

### Core Classes

**TrafficData**:
```python
@dataclass
class TrafficData:
    timestamp: datetime
    road_id: str
    location: Location
    traffic_volume: int
    average_speed: float
    occupancy_rate: float
    congestion_level: CongestionLevel
    road_type: RoadType
```

**LSTMTrafficPredictor**:
```python
class LSTMTrafficPredictor:
    def train(self, training_data, validation_data) -> None
    def predict(self, input_sequences) -> np.ndarray
    def save_model(self, filepath: str) -> None
    def load_model(self, filepath: str) -> None
```

**RouteOptimizer**:
```python
class RouteOptimizer:
    def find_optimal_routes(self, start_location, end_location, 
                          current_traffic_data) -> List[Route]
    def calculate_route_cost(self, route: Route) -> float
```

## Performance Optimization

### Model Training
- Use GPU acceleration when available
- Implement early stopping to prevent overfitting
- Use batch processing for large datasets
- Cache preprocessed features

### Visualization
- Enable marker clustering for large datasets
- Use data sampling for performance
- Implement progressive loading
- Cache map tiles

### Data Processing
- Use Parquet format for faster I/O
- Implement parallel processing
- Use streaming for large files
- Cache computed features

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Memory Issues**:
```python
# Reduce batch size
config.lstm_model.batch_size = 16

# Use data sampling
sampled_data = traffic_data[::10]  # Every 10th record
```

**Map Not Loading**:
- Check internet connection for map tiles
- Verify coordinate formats (latitude, longitude)
- Ensure data contains valid geographic coordinates

**Model Training Slow**:
- Enable GPU acceleration: `pip install tensorflow-gpu`
- Reduce sequence length or model complexity
- Use data sampling for initial experiments

### Performance Tuning

**For Large Datasets**:
```python
# Use data chunking
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Enable parallel processing
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(process_function, data_chunks)
```

**For Real-time Applications**:
```python
# Use streaming predictions
model.predict_stream(data_stream, batch_size=32)

# Implement caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def cached_prediction(input_hash):
    return model.predict(input_data)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `pytest tests/`
6. Submit pull request

### Code Style
- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings for all functions
- Include unit tests for new features

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **TensorFlow**: Machine learning framework
- **PyTorch Forecasting**: Temporal Fusion Transformer implementation
- **Streamlit**: Web application framework
- **Folium**: Interactive mapping
- **OSMnx**: Street network analysis
- **Plotly**: Interactive visualizations

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include error logs and system information

---

**Happy Traffic Predicting!**