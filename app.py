"""
Astana Traffic Congestion Prediction System
Main Streamlit Application - Modular Architecture
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.data.generator import create_demo_dataset
from config.settings import PAGE_CONFIG, CUSTOM_CSS

# Import all page modules
from pages.dashboard import dashboard_page
from pages.data_upload import data_upload_page
from pages.prediction import prediction_page
from pages.routing import routing_page
from pages.simulation import simulation_page
from pages.training import training_page
from pages.analytics import analytics_page


st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_demo_data():
    """Load Astana demo data."""
    try:
        traffic_df = pd.read_csv("data/samples/small_test/traffic_data.csv")
        weather_df = pd.read_csv("data/samples/small_test/weather_data.csv") 
        events_df = pd.read_csv("data/samples/small_test/events_data.csv")
        
        return {
            'traffic': traffic_df,
            'weather': weather_df,
            'events': events_df,
            'features': traffic_df
        }
    except FileNotFoundError:
        try:
            traffic_df = pd.read_csv("data/samples/astana_week/traffic_data.csv")
            weather_df = pd.read_csv("data/samples/astana_week/weather_data.csv") 
            events_df = pd.read_csv("data/samples/astana_week/events_data.csv")
            
            return {
                'traffic': traffic_df,
                'weather': weather_df,
                'events': events_df,
                'features': traffic_df
            }
        except FileNotFoundError:
            st.info("Generating Astana demo data... This may take a moment.")
            return create_demo_dataset()


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">Astana Traffic Congestion Prediction System</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time traffic analysis and prediction for Astana, Kazakhstan**")
    
    # Initialize session state
    if 'sim_time' not in st.session_state:
        st.session_state.sim_time = datetime.now()
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Data Upload", "Traffic Prediction", "Route Optimization", 
         "Real-time Simulation", "Model Training", "Analytics"],
        key="page_selector"
    )

    # Load demo data
    demo_data = load_demo_data()
    
    # Route to appropriate page module
    if page == "Dashboard":
        dashboard_page(demo_data)
    elif page == "Data Upload":
        data_upload_page()
    elif page == "Traffic Prediction":
        prediction_page(demo_data)
    elif page == "Route Optimization":
        routing_page()
    elif page == "Real-time Simulation":
        simulation_page(demo_data)
    elif page == "Model Training":
        training_page(demo_data)
    elif page == "Analytics":
        analytics_page(demo_data)


if __name__ == "__main__":
    main()
