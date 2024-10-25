"""
Temporal Fusion Transformer implementation for traffic congestion prediction.
Advanced time-series forecasting with attention mechanisms and interpretability.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    Baseline,
    QuantileLoss,
    NormalDistributionLoss
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFTTrafficPredictor:
    """Temporal Fusion Transformer for traffic congestion prediction."""
    
    def __init__(
        self,
        max_prediction_length: int = 6,  # Predict 6 hours ahead
        max_encoder_length: int = 24,    # Use 24 hours of history
        batch_size: int = 64,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 4,
        dropout: float = 0.3,
        hidden_continuous_size: int = 8,
        loss_function: str = 'quantile'  # 'quantile', 'normal', 'poisson'
    ):
        """
        Initialize TFT Traffic Predictor.
        
        Args:
            max_prediction_length: Maximum prediction horizon
            max_encoder_length: Maximum encoder length (lookback window)
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            hidden_size: Hidden size for the model
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
            loss_function: Loss function type
        """
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.loss_function = loss_function
        
        self.model = None
        self.trainer = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.dataset_parameters = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'congestion_level',
        time_idx_column: str = 'time_idx',
        group_ids: List[str] = ['road_id'],
        static_categoricals: List[str] = None,
        static_reals: List[str] = None,
        time_varying_known_categoricals: List[str] = None,
        time_varying_known_reals: List[str] = None,
        time_varying_unknown_categoricals: List[str] = None,
        time_varying_unknown_reals: List[str] = None,
        train_size: float = 0.7,
        val_size: float = 0.15
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for TFT training.
        
        Args:
            df: Input dataframe
            target_column: Target variable column name
            time_idx_column: Time index column name
            group_ids: List of group identifier columns
            static_categoricals: Static categorical features
            static_reals: Static real-valued features  
            time_varying_known_categoricals: Time-varying known categorical features
            time_varying_known_reals: Time-varying known real-valued features
            time_varying_unknown_categoricals: Time-varying unknown categorical features
            time_varying_unknown_reals: Time-varying unknown real-valued features
            train_size: Training data proportion
            val_size: Validation data proportion
            
        Returns:
            Tuple of (training_dataset, validation_dataset, test_dataset)
        """
        # Default feature categorization
        if static_categoricals is None:
            static_categoricals = ['road_type_highway', 'road_type_arterial', 'road_type_collector']
        
        if time_varying_known_categoricals is None:
            time_varying_known_categoricals = [
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour'
            ]
        
        if time_varying_known_reals is None:
            time_varying_known_reals = [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
            ]
        
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = [
                'traffic_volume', 'average_speed', 'occupancy_rate', 'speed_ratio',
                'temperature', 'humidity', 'precipitation', 'wind_speed'
            ]
        
        # Ensure required columns exist
        available_columns = df.columns.tolist()
        
        # Filter features to only include available columns
        static_categoricals = [col for col in static_categoricals if col in available_columns]
        time_varying_known_categoricals = [col for col in time_varying_known_categoricals if col in available_columns]
        time_varying_known_reals = [col for col in time_varying_known_reals if col in available_columns]
        time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in available_columns]
        
        # Create time index if not exists
        if time_idx_column not in df.columns:
            df = df.copy()
            df = df.sort_values(['road_id', 'timestamp'])
            df[time_idx_column] = df.groupby('road_id').cumcount()
        
        # Split data
        max_time_idx = df[time_idx_column].max()
        train_cutoff = int(max_time_idx * train_size)
        val_cutoff = int(max_time_idx * (train_size + val_size))
        
        # Create training dataset
        training_cutoff = df[time_idx_column] <= train_cutoff
        self.training_data = TimeSeriesDataSet(
            df[training_cutoff],
            time_idx=time_idx_column,
            target=target_column,
            group_ids=group_ids,
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals or [],
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals or [],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=group_ids, transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Create validation dataset
        validation_cutoff = (df[time_idx_column] > train_cutoff) & (df[time_idx_column] <= val_cutoff)
        self.validation_data = TimeSeriesDataSet.from_dataset(
            self.training_data,
            df[validation_cutoff],
            predict=True,
            stop_randomization=True
        )
        
        # Create test dataset
        test_cutoff = df[time_idx_column] > val_cutoff
        self.test_data = TimeSeriesDataSet.from_dataset(
            self.training_data,
            df[test_cutoff],
            predict=True,
            stop_randomization=True
        )
        
        # Store dataset parameters
        self.dataset_parameters = self.training_data.get_parameters()
        
        logger.info(f"Training samples: {len(self.training_data)}")
        logger.info(f"Validation samples: {len(self.validation_data)}")
        logger.info(f"Test samples: {len(self.test_data)}")
        
        return self.training_data, self.validation_data, self.test_data
    
    def create_model(self) -> TemporalFusionTransformer:
        """Create TFT model with specified configuration."""
        
        # Define loss function
        if self.loss_function == 'quantile':
            loss = QuantileLoss()
        elif self.loss_function == 'normal':
            loss = NormalDistributionLoss()
        elif self.loss_function == 'poisson':
            loss = PoissonLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=loss,
            log_interval=10,
            reduce_on_plateau_patience=4,
            optimizer="AdamW"
        )
        
        logger.info(f"Created TFT model with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train(
        self,
        max_epochs: int = 50,
        gpus: int = 0,
        gradient_clip_val: float = 0.1,
        early_stopping_patience: int = 10,
        log_dir: str = "logs",
        model_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the TFT model.
        
        Args:
            max_epochs: Maximum training epochs
            gpus: Number of GPUs to use
            gradient_clip_val: Gradient clipping value
            early_stopping_patience: Early stopping patience
            log_dir: Directory for logging
            model_save_path: Path to save the best model
            
        Returns:
            Training metrics and results
        """
        if self.training_data is None:
            raise ValueError("Training data must be prepared before training")
        
        if self.model is None:
            self.create_model()
        
        # Create data loaders
        train_dataloader = self.training_data.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_dataloader = self.validation_data.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=early_stopping_patience,
                verbose=True,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step")
        ]
        
        # Setup logger
        logger_tft = TensorBoardLogger(log_dir, name="tft_traffic")
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            gradient_clip_val=gradient_clip_val,
            callbacks=callbacks,
            logger=logger_tft,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True
        )
        
        # Train model
        logger.info("Starting TFT model training...")
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Get best model
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path and model_save_path:
            import shutil
            shutil.copy2(best_model_path, model_save_path)
            logger.info(f"Best model saved to {model_save_path}")
        
        # Calculate training metrics
        train_loss = self.trainer.callback_metrics.get('train_loss', None)
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        
        metrics = {
            'train_loss': float(train_loss) if train_loss else None,
            'val_loss': float(val_loss) if val_loss else None,
            'epochs_trained': self.trainer.current_epoch + 1,
            'best_model_path': best_model_path
        }
        
        logger.info("TFT training completed")
        return metrics
    
    def predict(
        self,
        dataset: TimeSeriesDataSet,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the trained model.
        
        Args:
            dataset: Dataset to predict on
            return_attention: Whether to return attention weights
            
        Returns:
            Predictions and optionally attention weights
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create data loader
        dataloader = dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )
        
        # Make predictions
        predictions = self.model.predict(
            dataloader,
            mode="prediction",
            return_attention=return_attention
        )
        
        return predictions
    
    def evaluate(self, dataset: TimeSeriesDataSet) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(dataset)
        
        # Convert to numpy for metric calculation
        if isinstance(predictions, dict):
            pred_values = predictions['prediction'].cpu().numpy()
        else:
            pred_values = predictions.cpu().numpy()
        
        # Get actual values
        dataloader = dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )
        
        actuals = []
        for batch in dataloader:
            actuals.append(batch[1].cpu().numpy())
        actual_values = np.concatenate(actuals, axis=0)
        
        # Calculate metrics
        if len(pred_values.shape) > 2:
            # For quantile predictions, use median
            pred_values = np.median(pred_values, axis=-1)
        
        # Flatten arrays for metric calculation
        pred_flat = pred_values.flatten()
        actual_flat = actual_values.flatten()
        
        # Remove NaN values
        valid_idx = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
        pred_clean = pred_flat[valid_idx]
        actual_clean = actual_flat[valid_idx]
        
        if len(pred_clean) == 0:
            logger.warning("No valid predictions for evaluation")
            return {}
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(actual_clean, pred_clean),
            'mae': mean_absolute_error(actual_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(actual_clean, pred_clean)),
            'r2_score': r2_score(actual_clean, pred_clean) if len(np.unique(actual_clean)) > 1 else 0.0,
            'smape': np.mean(2 * np.abs(pred_clean - actual_clean) / (np.abs(pred_clean) + np.abs(actual_clean) + 1e-8)) * 100
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def plot_predictions(
        self,
        dataset: TimeSeriesDataSet,
        idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            dataset: Dataset to plot predictions for
            idx: Index of the sample to plot
            save_path: Path to save the plot
        """
        predictions = self.predict(dataset, return_attention=False)
        
        # Get raw predictions
        raw_predictions = self.model.predict(
            dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0),
            mode="raw",
            return_index=True
        )
        
        # Plot the first sample
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Time series prediction plot
        if idx < len(raw_predictions.output):
            self.model.plot_prediction(
                raw_predictions.output[idx],
                raw_predictions.index[idx],
                add_loss_to_title=True,
                ax=axes[0, 0]
            )
            axes[0, 0].set_title(f"Prediction vs Actual - Sample {idx}")
        
        # Attention weights (if available)
        if hasattr(raw_predictions.output[idx], 'attention'):
            attention = raw_predictions.output[idx]['attention']
            if attention is not None:
                sns.heatmap(
                    attention[0].cpu().numpy(),
                    ax=axes[0, 1],
                    cmap='Blues'
                )
                axes[0, 1].set_title("Attention Weights")
        
        # Feature importance
        interpretation = self.model.interpret_output(
            raw_predictions.output[idx:idx+1],
            reduction="sum"
        )
        
        if 'attention' in interpretation:
            attention_by_variable = interpretation['attention'].cpu().numpy()
            importance_df = pd.DataFrame({
                'variable': self.model.hparams.reals + self.model.hparams.categoricals,
                'importance': attention_by_variable.mean(axis=0)
            }).sort_values('importance', ascending=True)
            
            importance_df.plot(
                x='variable',
                y='importance',
                kind='barh',
                ax=axes[1, 0]
            )
            axes[1, 0].set_title("Variable Importance")
        
        # Residuals plot
        predictions_np = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
        if hasattr(predictions_np, 'shape') and len(predictions_np.shape) > 2:
            predictions_np = np.median(predictions_np, axis=-1)
        
        # Get corresponding actuals
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        sample_batch = next(iter(dataloader))
        actual = sample_batch[1].cpu().numpy()
        
        if predictions_np.shape[0] > 0 and actual.shape[0] > 0:
            residuals = predictions_np[0] - actual[0]
            axes[1, 1].scatter(predictions_np[0], residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel("Predicted")
            axes[1, 1].set_ylabel("Residuals")
            axes[1, 1].set_title("Residuals Plot")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        torch.save(self.model.state_dict(), model_path)
        
        # Save model configuration and dataset parameters
        config = {
            'max_prediction_length': self.max_prediction_length,
            'max_encoder_length': self.max_encoder_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'attention_head_size': self.attention_head_size,
            'dropout': self.dropout,
            'hidden_continuous_size': self.hidden_continuous_size,
            'loss_function': self.loss_function,
            'dataset_parameters': self.dataset_parameters
        }
        
        config_path = model_path.replace('.pth', '_config.pkl')
        joblib.dump(config, config_path)
        
        logger.info(f"TFT model saved to {model_path}")
        logger.info(f"Model configuration saved to {config_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        # Load configuration
        config_path = model_path.replace('.pth', '_config.pkl')
        if Path(config_path).exists():
            config = joblib.load(config_path)
            
            # Restore configuration
            self.max_prediction_length = config['max_prediction_length']
            self.max_encoder_length = config['max_encoder_length']
            self.batch_size = config['batch_size']
            self.learning_rate = config['learning_rate']
            self.hidden_size = config['hidden_size']
            self.attention_head_size = config['attention_head_size']
            self.dropout = config['dropout']
            self.hidden_continuous_size = config['hidden_continuous_size']
            self.loss_function = config['loss_function']
            self.dataset_parameters = config['dataset_parameters']
            
            # Recreate model with saved parameters
            if self.dataset_parameters:
                # Create a dummy dataset to initialize the model
                dummy_dataset = TimeSeriesDataSet.from_parameters(
                    self.dataset_parameters,
                    pd.DataFrame()  # Empty dataframe, parameters contain the structure
                )
                self.training_data = dummy_dataset
                self.create_model()
                
                # Load model weights
                self.model.load_state_dict(torch.load(model_path))
                
                logger.info(f"TFT model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model configuration file not found: {config_path}")


def hyperparameter_tuning(
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    target_column: str = 'congestion_level',
    n_trials: int = 20,
    max_epochs: int = 30
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for TFT model.
    
    Args:
        df: Input dataframe
        param_grid: Dictionary of hyperparameters to tune
        target_column: Target variable column
        n_trials: Number of tuning trials
        max_epochs: Maximum epochs for each trial
        
    Returns:
        Best hyperparameters and performance metrics
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return {}
    
    def objective(trial):
        # Sample hyperparameters
        params = {}
        for param, values in param_grid.items():
            if isinstance(values[0], int):
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif isinstance(values[0], float):
                params[param] = trial.suggest_float(param, min(values), max(values))
            else:
                params[param] = trial.suggest_categorical(param, values)
        
        # Create and train model
        tft = TFTTrafficPredictor(**params)
        train_data, val_data, _ = tft.prepare_data(df, target_column=target_column)
        
        try:
            metrics = tft.train(max_epochs=max_epochs, gpus=0)
            val_loss = metrics.get('val_loss', float('inf'))
            return val_loss
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best validation loss: {best_value}")
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'study': study
    }