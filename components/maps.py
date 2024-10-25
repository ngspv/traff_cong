"""
Map visualization components using Folium.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from config.settings import CONGESTION_LABELS, CONGESTION_COLORS


def create_traffic_map(traffic_df):
    """Create traffic map visualization."""
    
    if traffic_df.empty:
        st.info("No traffic data available for mapping")
        return
    
    center_lat = traffic_df['latitude'].mean()
    center_lon = traffic_df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    for _, row in traffic_df.iterrows():
        color = CONGESTION_COLORS[int(row['congestion_level'])]
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"""
            Road: {row['road_id']}<br>
            Congestion: {CONGESTION_LABELS[int(row['congestion_level'])]}<br>
            Speed: {row['average_speed']:.1f} km/h<br>
            Volume: {row['traffic_volume']} veh/h
            """,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    st_folium(m, width=700, height=500)


def create_route_map(start_lat, start_lon, end_lat, end_lon, routes):
    """Create route visualization map."""
    
    center_lat = (start_lat + end_lat) / 2
    center_lon = (start_lon + end_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    folium.Marker(
        [start_lat, start_lon],
        popup="Start",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [end_lat, end_lon],
        popup="Destination", 
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, route in enumerate(routes):
        folium.PolyLine(
            locations=[[start_lat, start_lon], [end_lat, end_lon]],
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.8,
            popup=f"{route['route_id']}: {route['distance_km']:.1f} km, {route['estimated_time_min']:.0f} min"
        ).add_to(m)
    
    st_folium(m, width=700, height=500)


def create_live_traffic_map(traffic_df):
    """Create live traffic map for simulation."""
    create_traffic_map(traffic_df)
