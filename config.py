import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "traffic_db"
    username: str = "traffic_user"
    password: str = "traffic_pass"
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LSTMModelConfig:
    """LSTM model configuration."""
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2
    attention_units: int = 64
    use_attention: bool = True
    
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    sequence_length: int = 24  # Hours of historical data
    prediction_horizon: int = 6  # Hours to predict ahead
    
    model_save_path: str = str(MODELS_DIR / "lstm_traffic_model.h5")
    checkpoint_path: str = str(MODELS_DIR / "lstm_checkpoint.h5")
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]


@dataclass
class TFTModelConfig:
    """Temporal Fusion Transformer model configuration."""
    hidden_size: int = 32
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.2
    hidden_continuous_size: int = 16
    
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 0.03
    gradient_clip_val: float = 0.1
    
    max_encoder_length: int = 24  # Historical context
    max_prediction_length: int = 6  # Prediction horizon
    
    model_save_path: str = str(MODELS_DIR / "tft_traffic_model.ckpt")
    
    static_categoricals: List[str] = None
    static_reals: List[str] = None
    time_varying_known_categoricals: List[str] = None
    time_varying_known_reals: List[str] = None
    time_varying_unknown_categoricals: List[str] = None
    time_varying_unknown_reals: List[str] = None
    
    def __post_init__(self):
        if self.static_categoricals is None:
            self.static_categoricals = ['road_type', 'road_id']
        if self.static_reals is None:
            self.static_reals = ['latitude', 'longitude']
        if self.time_varying_known_categoricals is None:
            self.time_varying_known_categoricals = ['hour', 'day_of_week', 'month', 'is_weekend']
        if self.time_varying_known_reals is None:
            self.time_varying_known_reals = ['temperature', 'humidity', 'wind_speed']
        if self.time_varying_unknown_categoricals is None:
            self.time_varying_unknown_categoricals = ['weather_condition', 'event_type']
        if self.time_varying_unknown_reals is None:
            self.time_varying_unknown_reals = ['traffic_volume', 'average_speed', 'occupancy_rate']


@dataclass
class DataConfig:
    """Data processing and generation configuration."""
    traffic_data_source: str = "synthetic"  # "api", "database", "file", "synthetic"
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "your_weather_api_key")
    traffic_api_key: str = os.getenv("TRAFFIC_API_KEY", "your_traffic_api_key")
    
    feature_engineering_enabled: bool = True
    data_validation_enabled: bool = True
    outlier_detection_enabled: bool = True
    missing_data_threshold: float = 0.1  # Maximum proportion of missing data allowed
    
    num_roads: int = 50
    num_days: int = 30
    data_frequency_minutes: int = 15
    noise_level: float = 0.1
    
    raw_data_path: str = str(DATA_DIR / "raw")
    processed_data_path: str = str(DATA_DIR / "processed")
    synthetic_data_path: str = str(DATA_DIR / "synthetic")
    
    target_column: str = "congestion_level"
    datetime_column: str = "timestamp"
    location_columns: List[str] = None
    
    def __post_init__(self):
        if self.location_columns is None:
            self.location_columns = ['latitude', 'longitude']


@dataclass
class RoutingConfig:
    """Route optimization configuration."""
    network_type: str = "drive"  # "walk", "bike", "drive", "drive_service", "all"
    place_name: str = "Astana, Kazakhstan"  # Default location for routing
    custom_filter: str = None
    
    algorithm: str = "dijkstra"  # "dijkstra", "astar", "bellman_ford"
    weight_metric: str = "time"  # "time", "distance", "congestion_weighted"
    
    congestion_weight_multipliers: Dict[str, float] = None
    
    max_alternative_routes: int = 3
    route_diversity_factor: float = 0.3  # Higher values = more diverse routes
    
    max_network_size: int = 10000  # Maximum number of nodes in road network
    cache_network: bool = True
    network_cache_path: str = str(DATA_DIR / "road_networks")
    
    def __post_init__(self):
        if self.congestion_weight_multipliers is None:
            self.congestion_weight_multipliers = {
                "FREE_FLOW": 1.0,
                "LIGHT": 1.2,
                "MODERATE": 1.5,
                "HEAVY": 2.0,
                "SEVERE": 3.0
            }


@dataclass
class VisualizationConfig:
    """Visualization and UI configuration."""
    default_map_center: tuple = (51.1694, 71.4491)  # Astana coordinates
    default_zoom_level: int = 12
    map_tile_style: str = "OpenStreetMap"  # "OpenStreetMap", "CartoDB positron", "Stamen Terrain"
    
    congestion_color_scheme: Dict[str, str] = None
    road_type_color_scheme: Dict[str, str] = None
    
    chart_theme: str = "plotly_white"  # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"
    chart_color_palette: List[str] = None
    
    max_points_on_map: int = 1000
    clustering_enabled: bool = True
    heatmap_radius: int = 15
    
    def __post_init__(self):
        if self.congestion_color_scheme is None:
            self.congestion_color_scheme = {
                "FREE_FLOW": "#28a745",
                "LIGHT": "#ffc107",
                "MODERATE": "#fd7e14",
                "HEAVY": "#dc3545",
                "SEVERE": "#721c24"
            }
        
        if self.road_type_color_scheme is None:
            self.road_type_color_scheme = {
                "HIGHWAY": "#0066cc",
                "ARTERIAL": "#ff6600",
                "COLLECTOR": "#669900",
                "LOCAL": "#cc6600",
                "RESIDENTIAL": "#996633"
            }
        
        if self.chart_color_palette is None:
            self.chart_color_palette = [
                '#1f4e79', '#28a745', '#ffc107', '#fd7e14', '#dc3545',
                '#6f42c1', '#20c997', '#fd7e14', '#e83e8c', '#6c757d'
            ]


@dataclass
class StreamlitConfig:
    """Streamlit application configuration."""
    page_title: str = "Traffic Congestion Prediction System"
    page_icon: str = "🚗"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    max_upload_size_mb: int = 200
    allowed_file_types: List[str] = None
    
    cache_ttl_seconds: int = 3600  # 1 hour
    
    enable_caching: bool = True
    show_progress_bars: bool = True
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['csv', 'xlsx', 'json', 'parquet']


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    log_file_path: str = str(LOGS_DIR / "traffic_prediction.log")
    error_log_path: str = str(LOGS_DIR / "errors.log")
    
    max_log_size_mb: int = 10
    backup_count: int = 5
    
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    
    enable_authentication: bool = False
    session_timeout_minutes: int = 60
    
    anonymize_data: bool = True
    data_retention_days: int = 90
    
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.lstm_model = LSTMModelConfig()
        self.tft_model = TFTModelConfig()
        self.data = DataConfig()
        self.routing = RoutingConfig()
        self.visualization = VisualizationConfig()
        self.streamlit = StreamlitConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': self.database.__dict__,
            'lstm_model': self.lstm_model.__dict__,
            'tft_model': self.tft_model.__dict__,
            'data': self.data.__dict__,
            'routing': self.routing.__dict__,
            'visualization': self.visualization.__dict__,
            'streamlit': self.streamlit.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        
        for section_name, section_config in config_dict.items():
            if hasattr(config, section_name):
                section_obj = getattr(config, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return config
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


config = Config()

ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    config.logging.log_level = "WARNING"
    config.data.data_validation_enabled = True
    config.security.enable_authentication = True
elif ENV == "testing":
    config.logging.log_level = "DEBUG"
    config.data.num_roads = 10
    config.data.num_days = 7
elif ENV == "development":
    config.logging.log_level = "DEBUG"
    config.streamlit.show_progress_bars = True