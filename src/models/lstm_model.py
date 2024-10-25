"""
LSTM neural network implementation for traffic congestion prediction.
Handles sequential modeling of traffic patterns with proper training and evaluation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional, List
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMTrafficPredictor:
    """LSTM model for traffic congestion prediction."""
    
    def __init__(
        self,
        sequence_length: int = 24,
        lstm_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        model_type: str = 'regression'  # 'regression' or 'classification'
    ):
        """
        Initialize LSTM Traffic Predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
            model_type: 'regression' for continuous values, 'classification' for discrete levels
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        self.model = None
        self.history = None
        self.scaler = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(LSTM(
                units=units,
                return_sequences=return_sequences,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg)
            ))
            
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.model_type == 'classification':
            model.add(Dense(5, activation='softmax'))  # 5 congestion levels
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM training.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            save_path: Path to save the best model
            
        Returns:
            Training history and metrics
        """
        # Prepare sequence data
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        logger.info(f"Training sequences shape: {X_train_seq.shape}")
        logger.info(f"Validation sequences shape: {X_val_seq.shape}")
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        train_loss = self.model.evaluate(X_train_seq, y_train_seq, verbose=0)
        val_loss = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
        
        metrics = {
            'train_loss': train_loss[0] if isinstance(train_loss, list) else train_loss,
            'val_loss': val_loss[0] if isinstance(val_loss, list) else val_loss,
            'epochs_trained': len(self.history.history['loss']),
            'best_epoch': np.argmin(self.history.history['val_loss']) + 1
        }
        
        if self.model_type == 'classification':
            metrics['train_accuracy'] = train_loss[1]
            metrics['val_accuracy'] = val_loss[1]
        else:
            metrics['train_mae'] = train_loss[1]
            metrics['val_mae'] = val_loss[1]
        
        logger.info(f"Training completed. Best epoch: {metrics['best_epoch']}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_seq, _ = self.prepare_sequences(X, np.zeros(len(X)))
        predictions = self.model.predict(X_seq, verbose=0)
        
        if self.model_type == 'classification':
            return np.argmax(predictions, axis=1)
        else:
            return predictions.flatten()
    
    def predict_sequence(
        self,
        initial_sequence: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Predict multiple steps into the future.
        
        Args:
            initial_sequence: Initial sequence of shape (sequence_length, n_features)
            n_steps: Number of steps to predict
            
        Returns:
            Future predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Reshape for prediction
            seq_input = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred = self.model.predict(seq_input, verbose=0)
            
            if self.model_type == 'classification':
                pred_value = np.argmax(pred)
            else:
                pred_value = pred[0, 0]
            
            predictions.append(pred_value)
            
            # Update sequence (simplified - in practice, you'd update with actual features)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred_value  # Update first feature with prediction
        
        return np.array(predictions)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)
        predictions = self.model.predict(X_test_seq, verbose=0)
        
        if self.model_type == 'classification':
            pred_classes = np.argmax(predictions, axis=1)
            
            metrics = {
                'accuracy': np.mean(pred_classes == y_test_seq),
                'test_loss': self.model.evaluate(X_test_seq, y_test_seq, verbose=0)[0]
            }
            
            # Detailed classification metrics
            report = classification_report(
                y_test_seq, pred_classes, output_dict=True, zero_division=0
            )
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            })
            
        else:
            pred_values = predictions.flatten()
            
            metrics = {
                'mse': mean_squared_error(y_test_seq, pred_values),
                'mae': mean_absolute_error(y_test_seq, pred_values),
                'rmse': np.sqrt(mean_squared_error(y_test_seq, pred_values)),
                'r2_score': r2_score(y_test_seq, pred_values),
                'test_loss': self.model.evaluate(X_test_seq, y_test_seq, verbose=0)[0]
            }
        
        logger.info(f"Test evaluation metrics: {metrics}")
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metric plot
        if self.model_type == 'classification':
            metric_key = 'accuracy'
            metric_label = 'Accuracy'
        else:
            metric_key = 'mae'
            metric_label = 'MAE'
        
        if metric_key in self.history.history:
            axes[1].plot(self.history.history[metric_key], label=f'Training {metric_label}')
            axes[1].plot(self.history.history[f'val_{metric_key}'], label=f'Validation {metric_label}')
            axes[1].set_title(f'Model {metric_label}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_label)
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_samples: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Plot predictions vs actual values."""
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)
        predictions = self.model.predict(X_test_seq[:n_samples], verbose=0)
        
        if self.model_type == 'classification':
            pred_values = np.argmax(predictions, axis=1)
            actual_values = y_test_seq[:n_samples]
            
            plt.figure(figsize=(15, 8))
            
            # Time series plot
            plt.subplot(2, 1, 1)
            plt.plot(actual_values, label='Actual', alpha=0.7)
            plt.plot(pred_values, label='Predicted', alpha=0.7)
            plt.title('Traffic Congestion Level Predictions')
            plt.xlabel('Time Steps')
            plt.ylabel('Congestion Level')
            plt.legend()
            plt.grid(True)
            
            # Confusion matrix
            plt.subplot(2, 1, 2)
            cm = confusion_matrix(actual_values, pred_values)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
        else:
            pred_values = predictions.flatten()
            actual_values = y_test_seq[:n_samples]
            
            plt.figure(figsize=(15, 8))
            
            # Time series plot
            plt.subplot(2, 1, 1)
            plt.plot(actual_values, label='Actual', alpha=0.7)
            plt.plot(pred_values, label='Predicted', alpha=0.7)
            plt.title('Traffic Congestion Predictions')
            plt.xlabel('Time Steps')
            plt.ylabel('Congestion Level')
            plt.legend()
            plt.grid(True)
            
            # Scatter plot
            plt.subplot(2, 1, 2)
            plt.scatter(actual_values, pred_values, alpha=0.6)
            plt.plot([actual_values.min(), actual_values.max()], 
                     [actual_values.min(), actual_values.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Scatter Plot')
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        
        # Save model configuration
        config = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
            'model_type': self.model_type
        }
        
        config_path = model_path.replace('.h5', '_config.pkl')
        joblib.dump(config, config_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model configuration saved to {config_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        self.model = tf.keras.models.load_model(model_path)
        
        # Load model configuration
        config_path = model_path.replace('.h5', '_config.pkl')
        if Path(config_path).exists():
            config = joblib.load(config_path)
            self.sequence_length = config['sequence_length']
            self.lstm_units = config['lstm_units']
            self.dropout_rate = config['dropout_rate']
            self.l2_reg = config['l2_reg']
            self.learning_rate = config['learning_rate']
            self.model_type = config['model_type']
        
        logger.info(f"Model loaded from {model_path}")


def create_lstm_ensemble(
    models_config: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> List[LSTMTrafficPredictor]:
    """
    Create and train an ensemble of LSTM models.
    
    Args:
        models_config: List of model configurations
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        List of trained LSTM models
    """
    ensemble = []
    
    for i, config in enumerate(models_config):
        logger.info(f"Training ensemble model {i+1}/{len(models_config)}")
        
        model = LSTMTrafficPredictor(**config)
        model.train(X_train, y_train, X_val, y_val)
        ensemble.append(model)
    
    return ensemble


def ensemble_predict(
    ensemble: List[LSTMTrafficPredictor],
    X: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Make ensemble predictions.
    
    Args:
        ensemble: List of trained models
        X: Input features
        method: Ensemble method ('mean', 'median', 'mode')
        
    Returns:
        Ensemble predictions
    """
    predictions = [model.predict(X) for model in ensemble]
    predictions = np.array(predictions)
    
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    elif method == 'mode':
        from scipy import stats
        return stats.mode(predictions, axis=0)[0].flatten()
    else:
        raise ValueError(f"Unknown ensemble method: {method}")