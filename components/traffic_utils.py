"""
Traffic utility functions for data processing and generation.
"""

import streamlit as st
import numpy as np


def classify_astana_district(lat, lon):
    """Classify coordinates into Astana districts."""
    if 51.15 <= lat <= 51.19 and 71.43 <= lon <= 71.47:
        return "City Center"
    elif 51.10 <= lat <= 51.15 and 71.38 <= lon <= 71.42:
        return "Left Bank"
    elif 51.17 <= lat <= 51.20 and 71.44 <= lon <= 71.48:
        return "Right Bank"
    elif 51.12 <= lat <= 51.15 and 71.46 <= lon <= 71.49:
        return "Esil District"
    else:
        return "Suburban"


def generate_simulated_traffic(sim_time, base_traffic_df):
    """Generate simulated current traffic based on time."""
    
    # Initialize seed in session state if not present
    if 'traffic_seed' not in st.session_state:
        st.session_state.traffic_seed = 42
    
    # Use the session state seed for deterministic sampling
    sample_size = min(50, len(base_traffic_df))
    current_traffic = base_traffic_df.sample(n=sample_size, random_state=st.session_state.traffic_seed).copy()
    
    hour = sim_time.hour
    day_of_week = sim_time.weekday()
    
    if (7 <= hour <= 9) or (17 <= hour <= 19):  # Rush hours
        volume_multiplier = 1.5
        speed_multiplier = 0.7
    elif 22 <= hour or hour <= 6:  # Night
        volume_multiplier = 0.3
        speed_multiplier = 1.2
    else:
        volume_multiplier = 1.0
        speed_multiplier = 1.0
    
    if day_of_week >= 5:
        volume_multiplier *= 0.8
        speed_multiplier *= 1.1
    
    # Use seed for deterministic random values
    np.random.seed(st.session_state.traffic_seed)
    
    current_traffic['traffic_volume'] = (
        current_traffic['traffic_volume'] * volume_multiplier * 
        np.random.uniform(0.8, 1.2, len(current_traffic))
    ).astype(int)
    
    current_traffic['average_speed'] = (
        current_traffic['average_speed'] * speed_multiplier * 
        np.random.uniform(0.9, 1.1, len(current_traffic))
    )
    
    speed_ratios = current_traffic['average_speed'] / current_traffic['free_flow_speed']
    current_traffic['congestion_level'] = np.where(
        speed_ratios >= 0.8, 0,
        np.where(speed_ratios >= 0.6, 1,
                np.where(speed_ratios >= 0.4, 2,
                        np.where(speed_ratios >= 0.2, 3, 4)))
    )
    
    return current_traffic
