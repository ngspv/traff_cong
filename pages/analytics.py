"""
Analytics and Insights Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def analytics_page(demo_data):
    """Analytics and insights page."""
    
    st.header("Chart Traffic Analytics & Insights")
    
    traffic_df = demo_data['traffic']
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    
    st.subheader("Temporal Traffic Patterns")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Hourly Patterns", "Daily Patterns", "Weekly Patterns", "Congestion Trends"]
    )
    
    if analysis_type == "Hourly Patterns":
        hourly_stats = traffic_df.groupby(traffic_df['timestamp'].dt.hour).agg({
            'traffic_volume': 'mean',
            'average_speed': 'mean',
            'congestion_level': 'mean'
        }).round(2)
        
        fig = px.line(
            x=hourly_stats.index,
            y=hourly_stats['traffic_volume'],
            title="Average Traffic Volume by Hour",
            labels={'x': 'Hour of Day', 'y': 'Traffic Volume'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.line(
            x=hourly_stats.index,
            y=hourly_stats['average_speed'],
            title="Average Speed by Hour",
            labels={'x': 'Hour of Day', 'y': 'Average Speed (km/h)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    elif analysis_type == "Daily Patterns":
        traffic_df['day_of_week'] = traffic_df['timestamp'].dt.day_name()
        daily_stats = traffic_df.groupby('day_of_week').agg({
            'traffic_volume': 'mean',
            'average_speed': 'mean',
            'congestion_level': 'mean'
        }).round(2)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = daily_stats.reindex([day for day in day_order if day in daily_stats.index])
        
        fig = px.bar(
            x=daily_stats.index,
            y=daily_stats['traffic_volume'],
            title="Average Traffic Volume by Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Road Type Analysis")
    
    road_type_stats = traffic_df.groupby('road_type').agg({
        'traffic_volume': ['mean', 'std'],
        'average_speed': ['mean', 'std'],
        'congestion_level': 'mean'
    }).round(2)
    
    st.dataframe(road_type_stats)
    
    st.subheader("Congestion Heatmap")
    
    traffic_df['hour'] = traffic_df['timestamp'].dt.hour
    traffic_df['day'] = traffic_df['timestamp'].dt.day_name()
    
    heatmap_data = traffic_df.pivot_table(
        values='congestion_level',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(
            heatmap_data,
            title="Average Congestion Level by Day and Hour",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Congestion Level'},
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.subheader("Key Insights")
    
    insights = []
    
    hourly_congestion = traffic_df.groupby(traffic_df['timestamp'].dt.hour)['congestion_level'].mean()
    peak_hour = hourly_congestion.idxmax()
    peak_congestion = hourly_congestion.max()
    insights.append(f"Peak congestion occurs at {peak_hour}:00 with average level {peak_congestion:.2f}")
    
    road_congestion = traffic_df.groupby('road_id')['congestion_level'].mean()
    if len(road_congestion) > 0:
        best_road = road_congestion.idxmin()
        worst_road = road_congestion.idxmax()
        insights.append(f"Best road: {best_road} (avg congestion: {road_congestion[best_road]:.2f})")
        insights.append(f"Worst road: {worst_road} (avg congestion: {road_congestion[worst_road]:.2f})")
    
    speed_correlation = traffic_df['average_speed'].corr(traffic_df['traffic_volume'])
    insights.append(f"Speed-Volume correlation: {speed_correlation:.3f}")
    
    for insight in insights:
        st.info(insight)
