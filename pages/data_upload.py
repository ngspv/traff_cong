"""
Data Upload and Management Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.data.generator import TrafficDataGenerator
from src.data.pipeline import DataIngestionPipeline


def data_upload_page():
    """Data upload and management page."""
    
    st.header("Data Upload & Management")
    
    st.subheader("Upload Traffic Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload traffic data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} records loaded.")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Data Validation")
            
            required_columns = ['timestamp', 'latitude', 'longitude', 'road_id', 
                              'traffic_volume', 'average_speed']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
            else:
                st.success("All required columns present!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Quality Metrics:**")
                    st.write(f"- Total records: {len(df):,}")
                    st.write(f"- Missing values: {df.isnull().sum().sum():,}")
                    st.write(f"- Duplicate records: {df.duplicated().sum():,}")
                    st.write(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                with col2:
                    st.write("**Column Statistics:**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.dataframe(df[numeric_cols].describe())
                
                if st.button("Process and Save Data"):
                    pipeline = DataIngestionPipeline()
                    
                    try:
                        processed_df = pipeline.prepare_features(df)
                        
                        output_path = Path("data/processed/uploaded_data.csv")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        processed_df.to_csv(output_path, index=False)
                        
                        st.success(f"Data processed and saved to {output_path}")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.subheader("Generate Synthetic Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Start Date", datetime(2024, 1, 1))
        num_roads = st.slider("Number of Roads", 10, 100, 25)
    
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 1, 7))
        freq_hours = st.slider("Data Frequency (hours)", 1, 24, 1)
    
    with col3:
        events_per_day = st.slider("Events per Day", 0.0, 5.0, 1.0, 0.5)
    
    if st.button("Generate Synthetic Dataset"):
        with st.spinner("Generating data..."):
            generator = TrafficDataGenerator()
            
            datasets = generator.generate_complete_dataset(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                num_roads=num_roads,
                traffic_freq_minutes=freq_hours * 60,
                events_per_day=events_per_day,
                output_dir="data/generated"
            )
            
            st.success("Synthetic dataset generated successfully!")
            
            for name, df in datasets.items():
                st.write(f"**{name.title()}**: {len(df):,} records")
