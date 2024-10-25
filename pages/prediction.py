"""
Traffic Prediction Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta


def prediction_page(demo_data):
    """Traffic prediction page."""
    
    st.header("Traffic Congestion Prediction")
    
    st.subheader("Prediction Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_date = st.date_input(
            "Prediction Date",
            datetime.now().date() + timedelta(days=1)
        )
        prediction_time_val = st.time_input(
            "Prediction Time",
            datetime.now().time()
        )
        prediction_time = datetime.combine(prediction_date, prediction_time_val)
        
    with col2:
        road_id = st.selectbox(
            "Select Road",
            options=demo_data['traffic']['road_id'].unique()[:10]
        )
    
    with col3:
        horizon_hours = st.slider("Prediction Horizon (hours)", 1, 24, 6)
    
    st.subheader("Weather Conditions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature = st.slider("Temperature (°C)", -10, 40, 20)
    
    with col2:
        precipitation = st.slider("Precipitation (mm/h)", 0.0, 20.0, 0.0)
    
    with col3:
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
    
    with col4:
        visibility = st.slider("Visibility (km)", 1, 20, 15)
    
    st.subheader("Scheduled Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_type = st.selectbox(
            "Event Type",
            ["None", "Concert", "Sports", "Festival", "Construction", "Accident"]
        )
    
    with col2:
        if event_type != "None":
            expected_attendance = st.number_input("Expected Attendance", 0, 100000, 1000)
        else:
            expected_attendance = 0
    
    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Generating prediction..."):
            
            prediction_times = [
                prediction_time + timedelta(hours=i) for i in range(horizon_hours)
            ]
            
            base_congestion = 2  # Moderate
            
            weather_impact = 0
            if precipitation > 5:
                weather_impact += 1
            if temperature < 0:
                weather_impact += 0.5
            if visibility < 5:
                weather_impact += 0.5
            
            event_impact = 0
            if event_type != "None":
                event_impact = min(2, expected_attendance / 10000)
            
            predictions = []
            for i, pred_time in enumerate(prediction_times):
                hour = pred_time.hour
                
                if (7 <= hour <= 9) or (17 <= hour <= 19):
                    time_impact = 1
                elif 22 <= hour or hour <= 6:
                    time_impact = -1
                else:
                    time_impact = 0
                
                noise = np.random.normal(0, 0.3)
                
                pred_level = base_congestion + weather_impact + event_impact + time_impact + noise
                pred_level = max(0, min(4, round(pred_level)))
                
                predictions.append({
                    'time': pred_time,
                    'congestion_level': pred_level,
                    'confidence': max(0.6, 1.0 - abs(noise) * 0.5)
                })
            
            st.subheader("Prediction Results")
            
            pred_df = pd.DataFrame(predictions)
            congestion_labels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
            pred_df['congestion_label'] = pred_df['congestion_level'].map(
                lambda x: congestion_labels[int(x)]
            )
            
            fig = px.line(
                pred_df,
                x='time',
                y='congestion_level',
                title=f"Traffic Congestion Prediction for {road_id}",
                labels={'congestion_level': 'Congestion Level', 'time': 'Time'},
                color_discrete_sequence=['#1f4e79']
            )
            
            fig.update_layout(
                yaxis=dict(
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=congestion_labels
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_congestion = pred_df['congestion_level'].mean()
                st.metric("Average Predicted Congestion", 
                         congestion_labels[int(avg_congestion)])
            
            with col2:
                max_congestion = pred_df['congestion_level'].max()
                st.metric("Peak Congestion Level", 
                         congestion_labels[int(max_congestion)])
            
            with col3:
                avg_confidence = pred_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            st.subheader("Recommendations")
            
            if max_congestion >= 3:
                st.warning("Warning Heavy congestion expected. Consider alternative routes.")
            elif max_congestion >= 2:
                st.info("Info Moderate congestion expected. Allow extra travel time.")
            else:
                st.success("Check Light congestion expected. Normal travel conditions.")
