"""
Route Optimization Page
"""

import streamlit as st
import numpy as np
from datetime import datetime
from geopy.distance import geodesic

from src.data.schema import Location
from components.maps import create_route_map


def routing_page():
    """Route optimization page."""
    
    st.header("Route Optimization")
    
    st.subheader("Route Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Starting Location**")
        start_lat = st.number_input("Start Latitude", value=51.1694, format="%.6f", help="Default: Astana city center")
        start_lon = st.number_input("Start Longitude", value=71.4491, format="%.6f", help="Default: Astana city center")
    
    with col2:
        st.write("**Destination**")
        end_lat = st.number_input("End Latitude", value=51.1280, format="%.6f", help="Default: Astana airport area")
        end_lon = st.number_input("End Longitude", value=71.4678, format="%.6f", help="Default: Astana airport area")
    
    st.subheader("Route Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_weight = st.slider("Time Priority", 0.0, 1.0, 0.6, 0.1)
    
    with col2:
        distance_weight = st.slider("Distance Priority", 0.0, 1.0, 0.2, 0.1)
    
    with col3:
        comfort_weight = st.slider("Comfort Priority", 0.0, 1.0, 0.2, 0.1)
    
    departure_date = st.date_input(
        "Departure Date",
        datetime.now().date()
    )
    departure_time_val = st.time_input(
        "Departure Time",
        datetime.now().time()
    )
    departure_time = datetime.combine(departure_date, departure_time_val)
    
    num_routes = st.slider("Number of Alternative Routes", 1, 5, 3)
    
    if 'route_results' not in st.session_state:
        st.session_state.route_results = None
    if 'route_params' not in st.session_state:
        st.session_state.route_params = None
    
    col1, col2 = st.columns([1, 3])
    with col1:
        find_routes_clicked = st.button("Search Find Routes", type="primary")
    with col2:
        if st.session_state.route_results is not None:
            if st.button("Trash Clear Results"):
                st.session_state.route_results = None
                st.session_state.route_params = None
                st.rerun()
    
    if find_routes_clicked:
        with st.spinner("Finding optimal routes..."):
            current_params = {
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'time_weight': time_weight,
                'distance_weight': distance_weight,
                'comfort_weight': comfort_weight,
                'departure_time': departure_time,
                'num_routes': num_routes
            }
            st.session_state.route_params = current_params
            
            start_location = Location(start_lat, start_lon)
            end_location = Location(end_lat, end_lon)
            
            from geopy.distance import geodesic
            total_distance = geodesic(
                (start_lat, start_lon),
                (end_lat, end_lon)
            ).kilometers
            
            routes = []
            for i in range(num_routes):
                distance_factor = 1 + i * 0.15  # Each alternative is slightly longer
                time_factor = 1 + i * 0.1 + np.random.uniform(-0.05, 0.05)
                
                route = {
                    'route_id': f"Route {i+1}",
                    'distance_km': total_distance * distance_factor,
                    'estimated_time_min': (total_distance * distance_factor / 50) * 60 * time_factor,
                    'congestion_score': np.random.uniform(1, 4),
                    'route_type': np.random.choice(['Highway', 'Arterial', 'Mixed']),
                    'tolls': np.random.choice([0, 5, 10, 15]) if i > 0 else 0
                }
                
                normalized_time = route['estimated_time_min'] / 60  # Convert to hours
                normalized_distance = route['distance_km']
                normalized_comfort = 5 - route['congestion_score']
                
                route['total_score'] = (
                    time_weight * normalized_time +
                    distance_weight * normalized_distance +
                    comfort_weight * normalized_comfort
                )
                
                routes.append(route)
            
            routes.sort(key=lambda x: x['total_score'])
            
            st.session_state.route_results = routes
            st.success(f"Check Found {len(routes)} route options!")
    
    if st.session_state.route_results is not None:
        st.subheader("Road Route Options")
        
        if st.session_state.route_params:
            with st.expander("Info Route Parameters Used", expanded=False):
                params = st.session_state.route_params
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Start:** {params['start_lat']:.4f}, {params['start_lon']:.4f}")
                    st.write(f"**End:** {params['end_lat']:.4f}, {params['end_lon']:.4f}")
                with col2:
                    st.write(f"**Time Priority:** {params['time_weight']:.1f}")
                    st.write(f"**Distance Priority:** {params['distance_weight']:.1f}")
                    st.write(f"**Comfort Priority:** {params['comfort_weight']:.1f}")
                with col3:
                    st.write(f"**Departure:** {params['departure_time'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Routes Requested:** {params['num_routes']}")
        
        for i, route in enumerate(st.session_state.route_results):
            with st.expander(f"Road {route['route_id']} {'(Recommended)' if i == 0 else ''}", 
                           expanded=(i == 0)):
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Distance", f"{route['distance_km']:.1f} km")
                
                with col2:
                    st.metric("Estimated Time", f"{route['estimated_time_min']:.0f} min")
                
                with col3:
                    congestion_labels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
                    congestion_level = int(route['congestion_score'])
                    st.metric("Congestion", congestion_labels[congestion_level])
                
                with col4:
                    st.metric("Tolls", f"${route['tolls']}")
                
                st.write(f"**Route Type:** {route['route_type']}")
                st.write(f"**Overall Score:** {route['total_score']:.2f}")
                
                st.write("**Traffic Conditions:**")
                segments = 5
                segment_conditions = np.random.choice(
                    congestion_labels, segments, 
                    p=[0.2, 0.3, 0.3, 0.15, 0.05]
                )
                
                cols = st.columns(segments)
                for j, (col, condition) in enumerate(zip(cols, segment_conditions)):
                    color_map = {
                        "Free Flow": "Green",
                        "Light": "Yellow", 
                        "Moderate": "Orange",
                        "Heavy": "Red",
                        "Severe": "Purple"
                    }
                    col.write(f"Segment {j+1}: {color_map[condition]} {condition}")
        
        st.subheader("Route Visualization")
        if st.session_state.route_params:
            params = st.session_state.route_params
            create_route_map(
                params['start_lat'], params['start_lon'], 
                params['end_lat'], params['end_lon'], 
                st.session_state.route_results
            )
