"""
Real-time Traffic Simulation Page
"""

import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from components.traffic_utils import generate_simulated_traffic
from components.maps import create_live_traffic_map


def simulation_page(demo_data):
    """Real-time simulation page."""
    
    st.header("Real-time Traffic Simulation")
    
    if 'auto_update' not in st.session_state:
        st.session_state.auto_update = False
    
    mode_text = "Green **Auto-Update Mode:** Simulation updates automatically" if st.session_state.auto_update else "Lightbulb **Manual Control Mode:** Use the controls below to step through the simulation"
    st.info(mode_text)
    
    if 'sim_time' not in st.session_state:
        st.session_state.sim_time = datetime.now()
    
    st.subheader("Simulation Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_acceleration = st.slider("Time Step (minutes)", 1, 60, 10)
        weather_impact = st.checkbox("Weather Impact", value=True)
    
    with col2:
        include_events = st.checkbox("Include Random Events", value=True)
        update_interval = st.slider("Update Interval (seconds)", 1, 30, 5)
    
    with col3:
        st.write(f"**Current Time:**")
        st.write(f"**{st.session_state.sim_time.strftime('%H:%M:%S')}**")
        st.write(f"**{st.session_state.sim_time.strftime('%Y-%m-%d')}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.auto_update:
            if st.button("Pause Stop Auto-Update", type="secondary", use_container_width=True):
                st.session_state.auto_update = False
                st.rerun()
        else:
            if st.button("Play Start Auto-Update", type="primary", use_container_width=True):
                st.session_state.auto_update = True
                st.rerun()
    
    with col2:
        if st.button("Next Step Forward", use_container_width=True, disabled=st.session_state.auto_update):
            st.session_state.sim_time += timedelta(minutes=time_acceleration)
            st.session_state.traffic_seed = np.random.randint(0, 10000)
            st.session_state.sim_incidents = np.random.randint(0, 5)
            st.rerun()
    
    with col3:
        if st.button("Previous Step Backward", use_container_width=True, disabled=st.session_state.auto_update):
            st.session_state.sim_time -= timedelta(minutes=time_acceleration)
            st.session_state.traffic_seed = np.random.randint(0, 10000)
            st.session_state.sim_incidents = np.random.randint(0, 5)
            st.rerun()
    
    with col4:
        if st.button("Refresh Reset to Now", use_container_width=True):
            st.session_state.sim_time = datetime.now()
            st.session_state.traffic_seed = np.random.randint(0, 10000)
            st.session_state.sim_incidents = np.random.randint(0, 5)
            st.rerun()
    
    st.divider()
    
    st.subheader("Live Traffic Metrics")
    
    current_traffic = generate_simulated_traffic(st.session_state.sim_time, demo_data['traffic'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_speed = current_traffic['average_speed'].mean()
        st.metric("Average Speed", f"{avg_speed:.1f} km/h", 
                 delta=f"{np.random.uniform(-2, 2):.1f}")
    
    with col2:
        avg_volume = current_traffic['traffic_volume'].mean()
        st.metric("Average Volume", f"{avg_volume:.0f} veh/h",
                 delta=f"{np.random.randint(-50, 50)}")
    
    with col3:
        incidents = np.random.randint(0, 5)
        st.metric("Active Incidents", incidents)
    
    with col4:
        congested_roads = (current_traffic['congestion_level'] >= 3).sum()
        total_roads = len(current_traffic)
        congestion_pct = (congested_roads / total_roads) * 100
        st.metric("Congested Roads", f"{congestion_pct:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = px.histogram(
            current_traffic,
            x='traffic_volume',
            nbins=20,
            title="Current Traffic Volume Distribution"
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        fig_speed = px.scatter(
            current_traffic,
            x='average_speed',
            y='congestion_level',
            color='road_type',
            title="Speed vs Congestion Level"
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    st.divider()
    
    st.subheader("Live Traffic Map")
    
    create_live_traffic_map(current_traffic)
    
    if st.session_state.auto_update:
        import time
        time.sleep(update_interval)
        st.session_state.sim_time += timedelta(minutes=time_acceleration)
        st.session_state.traffic_seed = np.random.randint(0, 10000)
        st.session_state.sim_incidents = np.random.randint(0, 5)
        st.rerun()
