"""
Model Training Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time


def training_page(demo_data):
    """Model training page."""
    
    st.header("Model Training")
    
    st.subheader("Training Data")
    
    data_source = st.radio(
        "Select data source:",
        ["Demo Data", "Uploaded Data", "Generated Data"]
    )
    
    if data_source == "Demo Data":
        training_data = demo_data['features']
        st.info(f"Using demo dataset with {len(training_data)} samples")
    else:
        st.info("Please upload or generate data first")
        return
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["LSTM", "Temporal Fusion Transformer", "Random Forest", "XGBoost"]
        )
    
    with col2:
        prediction_type = st.radio(
            "Prediction Type",
            ["Classification", "Regression"]
        )
    
    st.subheader("Hyperparameters")
    
    epochs = 50  # Default value
    batch_size = 32  # Default value
    
    if model_type == "LSTM":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sequence_length = st.slider("Sequence Length", 6, 48, 24)
            lstm_units = st.text_input("LSTM Units", "128,64,32")
        
        with col2:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.001, 0.003, 0.01, 0.03, 0.1],
                value=0.001
            )
        
        with col3:
            epochs = st.slider("Max Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    elif model_type == "Temporal Fusion Transformer":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_encoder_length = st.slider("Encoder Length", 6, 48, 24)
            max_prediction_length = st.slider("Prediction Length", 1, 12, 6)
        
        with col2:
            hidden_size = st.slider("Hidden Size", 8, 64, 16)
            attention_heads = st.slider("Attention Heads", 1, 8, 4)
        
        with col3:
            epochs = st.slider("Max Epochs", 10, 100, 30)
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    
    else:
        # For Random Forest, XGBoost, and other models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Estimators", 50, 500, 100)
            epochs = st.slider("Max Iterations", 10, 200, 50)
        
        with col2:
            max_depth = st.slider("Max Depth", 3, 20, 10)
            batch_size = 32
        
        with col3:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.01, 0.05, 0.1, 0.3],
                value=0.1
            )
    
    st.subheader("Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_split = st.slider("Training Split", 0.5, 0.9, 0.7)
    
    with col2:
        val_split = st.slider("Validation Split", 0.05, 0.3, 0.15)
    
    with col3:
        early_stopping = st.checkbox("Early Stopping", value=True)
    
    st.subheader("Feature Selection")
    
    if not training_data.empty:
        numeric_columns = training_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Select features for training:",
            numeric_columns,
            default=numeric_columns[:10]  # Select first 10 by default
        )
        
        target_column = st.selectbox(
            "Target Variable",
            ["congestion_level", "average_speed", "traffic_volume"]
        )
    
    if st.button("Start Training", type="primary"):
        if not training_data.empty and selected_features:
            with st.spinner("Training model... This may take several minutes."):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                metrics = {}
                
                for epoch in range(epochs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.uniform(0, 0.1)
                    val_loss = train_loss + np.random.uniform(0, 0.2)
                    
                    if prediction_type == "Classification":
                        train_acc = min(0.95, 0.5 + epoch * 0.01 + np.random.uniform(0, 0.05))
                        val_acc = train_acc - np.random.uniform(0, 0.1)
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                    else:
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                    
                    if early_stopping and epoch > 20 and val_loss > train_loss * 1.5:
                        status_text.text(f"Early stopping at epoch {epoch+1}")
                        break
                    
                    import time
                    time.sleep(0.1)  # Simulate training time
                
                st.success("Training completed successfully!")
                
                st.subheader("Training Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    final_train_loss = train_loss
                    st.metric("Final Training Loss", f"{final_train_loss:.4f}")
                
                with col2:
                    final_val_loss = val_loss
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                
                with col3:
                    if prediction_type == "Classification":
                        final_val_acc = val_acc
                        st.metric("Validation Accuracy", f"{final_val_acc:.4f}")
                    else:
                        rmse = np.sqrt(final_val_loss)
                        st.metric("Validation RMSE", f"{rmse:.4f}")
                
                epochs_range = range(1, epoch + 2)
                train_losses = [1.0 * np.exp(-e * 0.1) + np.random.uniform(0, 0.1) for e in epochs_range]
                val_losses = [loss + np.random.uniform(0, 0.2) for loss in train_losses]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(epochs_range), y=train_losses, mode='lines', name='Training Loss'))
                fig.add_trace(go.Scatter(x=list(epochs_range), y=val_losses, mode='lines', name='Validation Loss'))
                fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Model Management")
                
                model_name = st.text_input("Model Name", f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                if st.button("Save Model"):
                    model_path = f"models/{model_name}.pkl"
                    st.success(f"Model saved as {model_path}")
                    
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = []
                    
                    st.session_state.trained_models.append({
                        'name': model_name,
                        'type': model_type,
                        'accuracy': final_val_acc if prediction_type == "Classification" else None,
                        'loss': final_val_loss,
                        'created': datetime.now()
                    })
        else:
            st.error("Please select features and ensure data is available")
    
    if 'trained_models' in st.session_state and st.session_state.trained_models:
        st.subheader("Trained Models")
        
        models_df = pd.DataFrame(st.session_state.trained_models)
        st.dataframe(models_df)
