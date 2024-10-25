"""
Dashboard Page - Main traffic monitoring dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import json

from components.traffic_utils import classify_astana_district


def dashboard_page(demo_data):
    """Main dashboard page for Astana traffic system."""
    
    st.header("Astana Traffic System Dashboard")
    st.markdown("**Real-time traffic monitoring for Astana districts**")
    
    st.subheader("Select Astana Dataset")
    
    available_datasets = {
        "Small Test": "data/samples/small_test",
        "Astana Week": "data/samples/astana_week", 
        "Left Bank Day": "data/samples/left_bank_day",
        "Right Bank Month": "data/samples/right_bank_month",
        "Highways Week": "data/samples/highways_week"
    }
    
    selected_dataset = st.selectbox(
        "Choose dataset to analyze:",
        list(available_datasets.keys()),
        index=0
    )
    
    try:
        dataset_path = available_datasets[selected_dataset]
        traffic_df = pd.read_csv(f"{dataset_path}/traffic_data.csv")
        weather_df = pd.read_csv(f"{dataset_path}/weather_data.csv")
        events_df = pd.read_csv(f"{dataset_path}/events_data.csv")
        
        try:
            import json
            with open(f"{dataset_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            st.info(f"Clipboard **{metadata['description']}** - {metadata['statistics']['traffic_records']} records")
        except:
            pass
            
    except FileNotFoundError:
        st.warning(f"Dataset {selected_dataset} not found. Using demo data.")
        traffic_df = demo_data['traffic']
        weather_df = demo_data['weather']
        events_df = demo_data['events']
    
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
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
    
    traffic_df['district'] = traffic_df.apply(
        lambda row: classify_astana_district(row['latitude'], row['longitude']), 
        axis=1
    )
    
    st.subheader("City Astana Districts Overview")
    
    district_cols = st.columns(len(traffic_df['district'].unique()))
    for i, district in enumerate(traffic_df['district'].unique()):
        district_data = traffic_df[traffic_df['district'] == district]
        avg_congestion = district_data['congestion_level'].mean()
        congestion_labels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
        
        with district_cols[i]:
            st.metric(
                f"Building {district}",
                congestion_labels[int(avg_congestion)] if not pd.isna(avg_congestion) else "N/A",
                f"{len(district_data)} roads"
            )
    
    st.subheader("Chart System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_roads = traffic_df['road_id'].nunique()
        st.metric("Road Roads Monitored", total_roads)
    
    with col2:
        avg_congestion = traffic_df['congestion_level'].mean()
        congestion_text = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"][int(avg_congestion)]
        congestion_color = ["Green", "Yellow", "Orange", "Red", "Black"][int(avg_congestion)]
        st.metric("TrafficLight Avg Congestion", f"{congestion_color} {congestion_text}")
    
    with col3:
        active_events = len(events_df)
        st.metric("Warning Active Events", active_events)
    
    with col4:
        avg_speed = traffic_df['average_speed'].mean()
        st.metric("Runner Average Speed", f"{avg_speed:.1f} km/h")
    
    tab1, tab2, tab3, tab4 = st.tabs(["TrafficLight Congestion", "Chart Trends", "Sun Weather Impact", "Pin Districts"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Congestion Distribution")
            congestion_counts = traffic_df['congestion_level'].value_counts().sort_index()
            congestion_labels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#721c24']
            
            fig_pie = px.pie(
                values=congestion_counts.values,
                names=[congestion_labels[i] for i in congestion_counts.index],
                color_discrete_sequence=colors,
                title="Traffic Congestion Levels in Astana"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Road Type Distribution") 
            road_type_counts = traffic_df['road_type'].value_counts()
            
            fig_road = px.bar(
                x=road_type_counts.index,
                y=road_type_counts.values,
                title="Roads by Type",
                labels={'x': 'Road Type', 'y': 'Count'},
                color=road_type_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_road, use_container_width=True)
    
    with tab2:
        st.subheader("Chart Traffic Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_traffic = traffic_df.groupby(traffic_df['timestamp'].dt.hour)['traffic_volume'].mean()
            
            fig_line = px.line(
                x=hourly_traffic.index,
                y=hourly_traffic.values,
                title="Average Traffic Volume by Hour",
                labels={'x': 'Hour of Day', 'y': 'Traffic Volume'},
                markers=True
            )
            fig_line.add_hline(y=hourly_traffic.mean(), line_dash="dash", 
                              annotation_text="Daily Average")
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            hourly_speed = traffic_df.groupby(traffic_df['timestamp'].dt.hour)['average_speed'].mean()
            
            fig_speed = px.line(
                x=hourly_speed.index,
                y=hourly_speed.values,
                title="Average Speed by Hour",
                labels={'x': 'Hour of Day', 'y': 'Speed (km/h)'},
                markers=True,
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_speed, use_container_width=True)
    
    with tab3:
        st.subheader("Weather Impact on Traffic")
        
        if not weather_df.empty:
            traffic_hourly = traffic_df.groupby(traffic_df['timestamp'].dt.floor('H')).agg({
                'average_speed': 'mean',
                'traffic_volume': 'mean',
                'congestion_level': 'mean'
            }).reset_index()
            
            weather_hourly = weather_df.groupby(weather_df['timestamp'].dt.floor('H')).agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'precipitation': 'mean'
            }).reset_index()
            
            merged_data = pd.merge(traffic_hourly, weather_hourly, on='timestamp', how='inner')
            
            if not merged_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_temp = px.scatter(
                        merged_data,
                        x='temperature',
                        y='average_speed', 
                        title="Temperature vs Average Speed",
                        labels={'temperature': 'Temperature (°C)', 'average_speed': 'Speed (km/h)'},
                        trendline="ols"
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with col2:
                    fig_precip = px.scatter(
                        merged_data,
                        x='precipitation',
                        y='congestion_level',
                        title="Precipitation vs Congestion",
                        labels={'precipitation': 'Precipitation (mm)', 'congestion_level': 'Congestion Level'},
                        trendline="ols"
                    )
                    st.plotly_chart(fig_precip, use_container_width=True)
        else:
            st.info("No weather data available for this dataset.")
    
    with tab4:
        st.subheader("Pin District-wise Analysis")
        
        district_analysis = traffic_df.groupby('district').agg({
            'traffic_volume': 'mean',
            'average_speed': 'mean',
            'congestion_level': 'mean',
            'road_id': 'nunique'
        }).round(2)
        district_analysis.columns = ['Avg Traffic Volume', 'Avg Speed (km/h)', 'Avg Congestion', 'Roads Count']
        
        st.dataframe(district_analysis, use_container_width=True)
        
        fig_district = px.bar(
            district_analysis.reset_index(),
            x='district',
            y='Avg Congestion',
            title="Average Congestion by Astana District",
            labels={'district': 'District', 'Avg Congestion': 'Congestion Level'},
            color='Avg Congestion',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_district, use_container_width=True)
    
    if not events_df.empty:
        st.subheader("Siren Current Events Impact")
        
        def classify_event_severity(row):
            if pd.isna(row['expected_attendance']):
                return 'low'
            attendance = row['expected_attendance']
            duration = row['duration_hours']
            
            if attendance > 15000 or duration > 6:
                return 'high'
            elif attendance > 5000 or duration > 3:
                return 'medium'
            else:
                return 'low'
        
        events_df['severity'] = events_df.apply(classify_event_severity, axis=1)
        
        event_severity = events_df['severity'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_events = px.bar(
                x=event_severity.index,
                y=event_severity.values,
                title="Events by Severity",
                labels={'x': 'Severity', 'y': 'Count'},
                color=event_severity.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_events, use_container_width=True)
        
        with col2:
            event_types = events_df['event_type'].value_counts().head(10)
            fig_types = px.pie(
                values=event_types.values,
                names=event_types.index,
                title="Top Event Types"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        st.subheader("Clipboard Recent Events")
        recent_events = events_df.sort_values('timestamp', ascending=False).head(10)
        display_events = recent_events[['timestamp', 'event_type', 'severity', 'expected_attendance', 'duration_hours']].copy()
        display_events['timestamp'] = display_events['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_events['expected_attendance'] = display_events['expected_attendance'].fillna(0).astype(int)
        display_events = display_events.rename(columns={
            'event_type': 'Event Type',
            'severity': 'Severity',
            'expected_attendance': 'Expected Attendance',
            'duration_hours': 'Duration (hrs)'
        })
        st.dataframe(display_events, use_container_width=True)
    
    st.subheader("Lightbulb District Insights")
    
    district_insights = {
        'City Center': "Historic center with government buildings. High congestion during business hours due to administrative activities.",
        'Left Bank': "Modern business district with Bayterek Tower. Heavy traffic during rush hours, many shopping centers.",
        'Right Bank': "Residential and commercial area. Moderate traffic, industrial zones create specific congestion patterns.",
        'Esil District': "Government quarter with presidential palace. Restricted access areas affect traffic flow patterns.",
        'Suburban': "Developing residential areas. Traffic mainly from commuters, lighter congestion overall."
    }
    
    present_districts = traffic_df['district'].unique()
    for district in present_districts:
        if district in district_insights:
            with st.expander(f"Info {district} - Click for insights"):
                st.write(district_insights[district])
                
                district_data = traffic_df[traffic_df['district'] == district]
                dist_col1, dist_col2, dist_col3 = st.columns(3)
                
                with dist_col1:
                    st.metric("Roads in District", len(district_data))
                with dist_col2:
                    avg_speed = district_data['average_speed'].mean()
                    st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
                with dist_col3:
                    congestion = district_data['congestion_level'].mean()
                    st.metric("Avg Congestion", f"{congestion:.1f}/4")
    
    st.subheader("Bullseye Traffic Management Recommendations")
    
    recommendations = []
    
    high_congestion_districts = traffic_df[traffic_df['congestion_level'] >= 3]['district'].value_counts()
    if not high_congestion_districts.empty:
        worst_district = high_congestion_districts.index[0]
        recommendations.append(f"Siren **Priority Alert**: {worst_district} shows high congestion. Consider traffic signal optimization.")
    
    if not events_df.empty:
        high_severity_events = len(events_df[events_df['severity'] == 'high'])
        if high_severity_events > 0:
            recommendations.append(f"Warning **Event Management**: {high_severity_events} high-severity events detected. Deploy traffic control units.")
    
    avg_speed = traffic_df['average_speed'].mean()
    if avg_speed < 25:
        recommendations.append("Snail **Speed Alert**: Average speed is low. Consider alternative route suggestions.")
    
    if not weather_df.empty:
        recent_weather = weather_df.iloc[-1]
        if recent_weather['precipitation'] > 5:
            recommendations.append("Rain **Weather Advisory**: Heavy precipitation detected. Increase following distances and reduce speed limits.")
        if recent_weather['temperature'] < 0:
            recommendations.append("Snowflake **Winter Conditions**: Sub-zero temperatures. Monitor for ice formation on major routes.")
    
    if not recommendations:
        recommendations = [
            "Check **Normal Conditions**: Traffic flow appears normal across Astana.",
            "Search **Monitoring**: Continue regular monitoring of key intersection points.",
            "Chart **Data Collection**: Maintain real-time data collection for predictive analytics."
        ]
    
    for rec in recommendations:
        st.write(rec)
    
    st.subheader("Map Traffic Map - Astana")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("Lightbulb **Tip**: Use the refresh button to update map data. Switch to static view if the map reloads frequently.")
    with col2:
        if st.button("Refresh Refresh Map"):
            if 'map_key' not in st.session_state:
                st.session_state.map_key = 0
            st.session_state.map_key += 1
    with col3:
        use_static_map = st.checkbox("Chart Static View", help="Use static map to prevent reloading")
    
    if 'map_key' not in st.session_state:
        st.session_state.map_key = 0
    
    if use_static_map:
        traffic_sample = traffic_df.sample(min(50, len(traffic_df)), random_state=42 + st.session_state.map_key)
        
        fig_map = px.scatter_map(
            traffic_sample,
            lat='latitude',
            lon='longitude',
            color='congestion_level',
            size='traffic_volume',
            hover_data=['road_id', 'district', 'average_speed'],
            color_continuous_scale=['green', 'yellow', 'orange', 'red', 'darkred'],
            map_style='open-street-map',
            zoom=10,
            center={'lat': 51.1694, 'lon': 71.4491},
            title="Traffic Congestion Map - Astana",
            height=500
        )
        
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    else:
        @st.cache_data(ttl=60)  # Cache for 1 minute
        def create_traffic_map_data(traffic_sample_size=25, map_key=0):
            """Create stable map data that doesn't change on every interaction."""
            traffic_sample = traffic_df.sample(min(traffic_sample_size, len(traffic_df)), random_state=42 + map_key)
            return traffic_sample
        
        traffic_sample = create_traffic_map_data(25, st.session_state.map_key)
        events_sample = events_df.head(5) if not events_df.empty else events_df
        
        astana_center = [51.1694, 71.4491]
        m = folium.Map(
            location=astana_center, 
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        for _, row in traffic_sample.iterrows():
            colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            color = colors[int(row['congestion_level'])]
            
            popup_text = f"""
            <b>Road ID:</b> {row['road_id']}<br>
            <b>District:</b> {row['district']}<br>
            <b>Speed:</b> {row['average_speed']:.1f} km/h<br>
            <b>Volume:</b> {row['traffic_volume']:.0f}<br>
            <b>Congestion:</b> {["Free Flow", "Light", "Moderate", "Heavy", "Severe"][int(row['congestion_level'])]}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=popup_text,
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        if not events_sample.empty:
            for _, event in events_sample.iterrows():
                icon_color = 'red' if event['severity'] == 'high' else 'orange' if event['severity'] == 'medium' else 'blue'
                attendance_text = f"Expected: {int(event['expected_attendance']):,}" if pd.notna(event['expected_attendance']) else "No attendance data"
                popup_text = f"""
                <b>{event['event_type']}</b><br>
                Severity: {event['severity']}<br>
                {attendance_text}<br>
                Duration: {event['duration_hours']:.1f} hours
                """
                folium.Marker(
                    location=[event['latitude'], event['longitude']],
                    popup=popup_text,
                    icon=folium.Icon(color=icon_color, icon='exclamation-sign')
                ).add_to(m)
        
        try:
            map_data = st_folium(
                m, 
                key="traffic_map_stable",
                width=700,
                height=450,
                returned_objects=["last_object_clicked"]
            )
        except Exception as e:
            st.error(f"Map display error: {e}")
            st.info("The map is temporarily unavailable. Please try the 'Static View' option above.")
