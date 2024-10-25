"""
Logging configuration for the Traffic Congestion Prediction System.
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    if log_file is None:
        log_file = log_dir / "traffic_prediction.log"
    
    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': log_level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_file),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'error_file': {
                'level': 'ERROR',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_dir / "errors.log"),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 3,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file', 'error_file'] if console_output else ['file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'traffic_prediction': {
                'handlers': ['default', 'file'] if console_output else ['file'],
                'level': log_level,
                'propagate': False
            },
            'streamlit': {
                'handlers': ['default'] if console_output else [],
                'level': 'WARNING',
                'propagate': False
            },
            'matplotlib': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            },
            'PIL': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    # Remove console handler if not needed
    if not console_output:
        handlers = logging_config['loggers']['']['handlers']
        if 'default' in handlers:
            handlers.remove('default')
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to caller's module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'traffic_prediction')
    
    return logging.getLogger(name)


# Custom log filters
class DebugFilter(logging.Filter):
    """Filter to only allow debug messages."""
    
    def filter(self, record):
        return record.levelno == logging.DEBUG


class ProductionFilter(logging.Filter):
    """Filter to exclude debug and info messages in production."""
    
    def filter(self, record):
        return record.levelno >= logging.WARNING


# Context manager for temporary log level changes
class LogLevel:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger_name: str = None, level: str = "DEBUG"):
        self.logger_name = logger_name or 'traffic_prediction'
        self.level = getattr(logging, level.upper())
        self.original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(self.level)
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.original_level)


# Performance timing decorator
def log_execution_time(logger_name: str = None):
    """Decorator to log execution time of functions."""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Initialize logging when module is imported
def init_logging():
    """Initialize logging with default settings."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Adjust settings based on environment
    if environment == "production":
        console_output = False
        log_level = "WARNING"
    elif environment == "testing":
        console_output = True
        log_level = "DEBUG"
    else:  # development
        console_output = True
        log_level = "INFO"
    
    setup_logging(log_level=log_level, console_output=console_output)


# Auto-initialize when imported
if not logging.getLogger().handlers:
    init_logging()