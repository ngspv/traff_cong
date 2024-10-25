"""
Advanced map visualization module using Folium and other mapping libraries.
Provides interactive maps for traffic patterns, routes, and congestion analysis.
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import json
import branca.colormap as cm
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.schema import Location, TrafficData, CongestionLevel, RoadType
from ..utils.routing import Route, Road


class TrafficMapVisualizer:
    """Advanced traffic map visualization with Folium."""
    
    def __init__(self, default_location: Tuple[float, float] = (51.1694, 71.4491)):
        """
        Initialize the map visualizer.
        
        Args:
            default_location: Default center location (lat, lon) - Astana, Kazakhstan
        """
        self.default_location = default_location
        self.congestion_colors = {
            CongestionLevel.FREE_FLOW: '#28a745',
            CongestionLevel.LIGHT: '#ffc107',
            CongestionLevel.MODERATE: '#fd7e14',
            CongestionLevel.HEAVY: '#dc3545',
            CongestionLevel.SEVERE: '#721c24'
        }
        
        self.road_type_colors = {
            RoadType.HIGHWAY: '#0066cc',
            RoadType.ARTERIAL: '#ff6600',
            RoadType.COLLECTOR: '#669900',
            RoadType.LOCAL: '#cc6600',
            RoadType.RESIDENTIAL: '#996633'
        }
    
    def create_base_map(
        self,
        center_location: Optional[Tuple[float, float]] = None,
        zoom_start: int = 12,
        tiles: str = 'OpenStreetMap'
    ) -> folium.Map:
        """
        Create a base map with standard settings.
        
        Args:
            center_location: Center location for the map
            zoom_start: Initial zoom level
            tiles: Map tile style
            
        Returns:
            Folium map object
        """
        if center_location is None:
            center_location = self.default_location
        
        m = folium.Map(
            location=center_location,
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def add_traffic_heatmap(
        self,
        map_obj: folium.Map,
        traffic_data: List[TrafficData],
        metric: str = 'congestion_level',
        radius: int = 15
    ) -> folium.Map:
        """
        Add traffic heatmap layer to the map.
        
        Args:
            map_obj: Folium map object
            traffic_data: List of traffic data points
            metric: Metric to visualize ('congestion_level', 'traffic_volume', 'occupancy_rate')
            radius: Heatmap point radius
            
        Returns:
            Updated map object
        """
        if not traffic_data:
            return map_obj
        
        heat_data = []
        for traffic in traffic_data:
            lat = traffic.location.latitude
            lon = traffic.location.longitude
            
            if metric == 'congestion_level':
                intensity = traffic.congestion_level.value / 4.0  # Normalize to 0-1
            elif metric == 'traffic_volume':
                max_volume = max(t.traffic_volume for t in traffic_data)
                intensity = traffic.traffic_volume / max_volume if max_volume > 0 else 0
            elif metric == 'occupancy_rate':
                intensity = traffic.occupancy_rate / 100.0
            else:
                intensity = 0.5
            
            heat_data.append([lat, lon, intensity])
        
        heatmap = plugins.HeatMap(
            heat_data,
            name=f'{metric.title()} Heatmap',
            radius=radius,
            blur=10,
            max_zoom=18,
            gradient={
                0.0: 'blue',
                0.3: 'cyan', 
                0.5: 'lime',
                0.7: 'yellow',
                1.0: 'red'
            }
        )
        heatmap.add_to(map_obj)
        
        return map_obj
    
    def add_traffic_markers(
        self,
        map_obj: folium.Map,
        traffic_data: List[TrafficData],
        cluster: bool = True,
        show_details: bool = True
    ) -> folium.Map:
        """
        Add traffic markers to the map.
        
        Args:
            map_obj: Folium map object
            traffic_data: List of traffic data points
            cluster: Whether to cluster nearby markers
            show_details: Whether to show detailed popups
            
        Returns:
            Updated map object
        """
        if not traffic_data:
            return map_obj
        
        if cluster:
            marker_cluster = plugins.MarkerCluster(name='Traffic Points')
            marker_cluster.add_to(map_obj)
            container = marker_cluster
        else:
            container = map_obj
        
        for traffic in traffic_data:
            color = self.congestion_colors.get(
                traffic.congestion_level, '#gray'
            )
            
            if show_details:
                popup_content = f"""
                <div style="font-family: Arial, sans-serif; width: 200px;">
                    <h4>{traffic.road_id}</h4>
                    <hr>
                    <b>Congestion:</b> {traffic.congestion_level.name}<br>
                    <b>Speed:</b> {traffic.average_speed:.1f} km/h<br>
                    <b>Volume:</b> {traffic.traffic_volume:,} veh/h<br>
                    <b>Occupancy:</b> {traffic.occupancy_rate:.1f}%<br>
                    <b>Road Type:</b> {traffic.road_type.value.title()}<br>
                    <b>Time:</b> {traffic.timestamp.strftime('%Y-%m-%d %H:%M')}
                </div>
                """
            else:
                popup_content = f"{traffic.road_id}: {traffic.congestion_level.name}"
            
            folium.CircleMarker(
                location=[traffic.location.latitude, traffic.location.longitude],
                radius=8,
                popup=folium.Popup(popup_content, max_width=250),
                color='white',
                weight=2,
                fillColor=color,
                fillOpacity=0.8,
                tooltip=f"Road {traffic.road_id}"
            ).add_to(container)
        
        return map_obj
    
    def add_route_visualization(
        self,
        map_obj: folium.Map,
        routes: List[Route],
        show_alternatives: bool = True,
        animate: bool = False
    ) -> folium.Map:
        """
        Add route visualization to the map.
        
        Args:
            map_obj: Folium map object
            routes: List of routes to visualize
            show_alternatives: Whether to show alternative routes
            animate: Whether to animate the routes
            
        Returns:
            Updated map object
        """
        if not routes:
            return map_obj
        
        route_colors = ['#0066cc', '#ff0000', '#00cc00', '#ff6600', '#9900cc']
        
        for i, route in enumerate(routes):
            if i == 0 or show_alternatives:
                color = route_colors[i % len(route_colors)]
                weight = 6 if i == 0 else 4
                opacity = 0.8 if i == 0 else 0.6
                
                coordinates = []
                for road in route.roads:
                    coordinates.append([
                        road.start_location.latitude,
                        road.start_location.longitude
                    ])
                
                if route.roads:
                    coordinates.append([
                        route.roads[-1].end_location.latitude,
                        route.roads[-1].end_location.longitude
                    ])
                
                route_line = folium.PolyLine(
                    locations=coordinates,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    popup=f"""
                    <b>Route {i+1}</b><br>
                    Distance: {route.total_distance:.1f} km<br>
                    Time: {route.total_time:.0f} min<br>
                    Cost: {route.total_cost:.2f}
                    """,
                    tooltip=f"Route {i+1}: {route.total_distance:.1f} km"
                )
                route_line.add_to(map_obj)
                
                if i == 0:
                    folium.Marker(
                        location=[route.start_location.latitude, route.start_location.longitude],
                        icon=folium.Icon(color='green', icon='play'),
                        popup="Start"
                    ).add_to(map_obj)
                    
                    folium.Marker(
                        location=[route.end_location.latitude, route.end_location.longitude],
                        icon=folium.Icon(color='red', icon='stop'),
                        popup="Destination"
                    ).add_to(map_obj)
                
                if animate and i == 0:
                    self._add_route_animation(map_obj, coordinates)
        
        return map_obj
    
    def _add_route_animation(
        self,
        map_obj: folium.Map,
        coordinates: List[List[float]]
    ) -> None:
        """Add animated marker along route."""
        animation_points = []
        for coord in coordinates:
            animation_points.append(coord)
        
        plugins.AntPath(
            locations=coordinates,
            dash_array=[10, 20],
            delay=1000,
            color='blue',
            pulse_color='white'
        ).add_to(map_obj)
    
    def add_congestion_zones(
        self,
        map_obj: folium.Map,
        traffic_data: List[TrafficData],
        zone_radius: float = 1.0
    ) -> folium.Map:
        """
        Add congestion zones to the map.
        
        Args:
            map_obj: Folium map object
            traffic_data: List of traffic data points
            zone_radius: Radius of congestion zones in km
            
        Returns:
            Updated map object
        """
        if not traffic_data:
            return map_obj
        
        congestion_groups = {}
        for traffic in traffic_data:
            level = traffic.congestion_level
            if level not in congestion_groups:
                congestion_groups[level] = []
            congestion_groups[level].append(traffic)
        
        for level, traffic_points in congestion_groups.items():
            if level in [CongestionLevel.HEAVY, CongestionLevel.SEVERE]:
                color = self.congestion_colors[level]
                
                for traffic in traffic_points:
                    folium.Circle(
                        location=[traffic.location.latitude, traffic.location.longitude],
                        radius=zone_radius * 1000,  # Convert to meters
                        color=color,
                        fillColor=color,
                        fillOpacity=0.3,
                        popup=f"Congestion Zone: {level.name}",
                        tooltip=f"{level.name} congestion area"
                    ).add_to(map_obj)
        
        return map_obj
    
    def add_traffic_flow_arrows(
        self,
        map_obj: folium.Map,
        roads: List[Road],
        flow_threshold: int = 500
    ) -> folium.Map:
        """
        Add traffic flow direction arrows.
        
        Args:
            map_obj: Folium map object
            roads: List of road segments
            flow_threshold: Minimum traffic volume to show arrows
            
        Returns:
            Updated map object
        """
        for road in roads:
            if hasattr(road, 'traffic_volume') and road.traffic_volume >= flow_threshold:
                start_lat = road.start_location.latitude
                start_lon = road.start_location.longitude
                end_lat = road.end_location.latitude
                end_lon = road.end_location.longitude
                
                mid_lat = (start_lat + end_lat) / 2
                mid_lon = (start_lon + end_lon) / 2
                
                bearing = self._calculate_bearing(
                    (start_lat, start_lon),
                    (end_lat, end_lon)
                )
                
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.Icon(
                        icon='arrow-up',
                        color='blue',
                        angle=bearing
                    ),
                    tooltip=f"Traffic flow: {road.traffic_volume} veh/h"
                ).add_to(map_obj)
        
        return map_obj
    
    def _calculate_bearing(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float]
    ) -> float:
        """Calculate bearing between two points."""
        start_lat, start_lon = np.radians(start_point)
        end_lat, end_lon = np.radians(end_point)
        
        d_lon = end_lon - start_lon
        
        x = np.sin(d_lon) * np.cos(end_lat)
        y = (np.cos(start_lat) * np.sin(end_lat) - 
             np.sin(start_lat) * np.cos(end_lat) * np.cos(d_lon))
        
        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def add_temporal_slider(
        self,
        map_obj: folium.Map,
        traffic_data_by_time: Dict[datetime, List[TrafficData]]
    ) -> folium.Map:
        """
        Add temporal slider for time-based traffic visualization.
        
        Args:
            map_obj: Folium map object
            traffic_data_by_time: Dictionary mapping timestamps to traffic data
            
        Returns:
            Updated map object
        """
        if not traffic_data_by_time:
            return map_obj
        
        features = []
        
        for timestamp, traffic_data in traffic_data_by_time.items():
            for traffic in traffic_data:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            traffic.location.longitude,
                            traffic.location.latitude
                        ]
                    },
                    "properties": {
                        "time": timestamp.isoformat(),
                        "popup": f"Road: {traffic.road_id}",
                        "style": {
                            "color": self.congestion_colors[traffic.congestion_level],
                            "fillOpacity": 0.8
                        }
                    }
                }
                features.append(feature)
        
        timestamped_geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        plugins.TimestampedGeoJson(
            timestamped_geojson,
            period="PT1H",  # 1 hour intervals
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm:ss",
            time_slider_drag_update=True
        ).add_to(map_obj)
        
        return map_obj
    
    def create_comparison_map(
        self,
        traffic_data_before: List[TrafficData],
        traffic_data_after: List[TrafficData],
        title: str = "Traffic Comparison"
    ) -> folium.Map:
        """
        Create a dual-pane comparison map.
        
        Args:
            traffic_data_before: Traffic data for left pane
            traffic_data_after: Traffic data for right pane
            title: Map title
            
        Returns:
            Folium map with side-by-side comparison
        """
        all_traffic = traffic_data_before + traffic_data_after
        if all_traffic:
            center_lat = np.mean([t.location.latitude for t in all_traffic])
            center_lon = np.mean([t.location.longitude for t in all_traffic])
        else:
            center_lat, center_lon = self.default_location
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12
        )
        
        plugins.SideBySideLayers().add_to(m)
        
        before_layer = folium.FeatureGroup(name="Before")
        self._add_traffic_points_to_layer(before_layer, traffic_data_before)
        before_layer.add_to(m)
        
        after_layer = folium.FeatureGroup(name="After")
        self._add_traffic_points_to_layer(after_layer, traffic_data_after)
        after_layer.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def _add_traffic_points_to_layer(
        self,
        layer: folium.FeatureGroup,
        traffic_data: List[TrafficData]
    ) -> None:
        """Add traffic points to a specific layer."""
        for traffic in traffic_data:
            color = self.congestion_colors.get(traffic.congestion_level, '#gray')
            
            folium.CircleMarker(
                location=[traffic.location.latitude, traffic.location.longitude],
                radius=6,
                color='white',
                weight=1,
                fillColor=color,
                fillOpacity=0.8,
                popup=f"""
                Road: {traffic.road_id}<br>
                Congestion: {traffic.congestion_level.name}<br>
                Speed: {traffic.average_speed:.1f} km/h
                """
            ).add_to(layer)
    
    def add_legend(self, map_obj: folium.Map, legend_type: str = "congestion") -> folium.Map:
        """
        Add a legend to the map.
        
        Args:
            map_obj: Folium map object
            legend_type: Type of legend ('congestion', 'road_type')
            
        Returns:
            Updated map object
        """
        if legend_type == "congestion":
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Congestion Levels</b></p>
            <p><i class="fa fa-circle" style="color:#28a745"></i> Free Flow</p>
            <p><i class="fa fa-circle" style="color:#ffc107"></i> Light</p>
            <p><i class="fa fa-circle" style="color:#fd7e14"></i> Moderate</p>
            <p><i class="fa fa-circle" style="color:#dc3545"></i> Heavy</p>
            <p><i class="fa fa-circle" style="color:#721c24"></i> Severe</p>
            </div>
            '''
        elif legend_type == "road_type":
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Road Types</b></p>
            <p><i class="fa fa-circle" style="color:#0066cc"></i> Highway</p>
            <p><i class="fa fa-circle" style="color:#ff6600"></i> Arterial</p>
            <p><i class="fa fa-circle" style="color:#669900"></i> Collector</p>
            <p><i class="fa fa-circle" style="color:#cc6600"></i> Local</p>
            <p><i class="fa fa-circle" style="color:#996633"></i> Residential</p>
            </div>
            '''
        else:
            return map_obj
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
        return map_obj
    
    def export_map(
        self,
        map_obj: folium.Map,
        filename: str,
        format: str = "html"
    ) -> None:
        """
        Export map to file.
        
        Args:
            map_obj: Folium map object
            filename: Output filename
            format: Export format ('html', 'png')
        """
        if format == "html":
            map_obj.save(filename)
        elif format == "png":
            try:
                import io
                from PIL import Image
                import base64
                
                map_obj.save("temp_map.html")
                
                print(f"PNG export requires additional setup. Saved as HTML instead: {filename}")
                
            except ImportError:
                print("PIL not available for PNG export. Saved as HTML instead.")
                map_obj.save(filename.replace('.png', '.html'))


class TrafficChartVisualizer:
    """Create charts and plots for traffic analysis."""
    
    def __init__(self):
        """Initialize the chart visualizer."""
        self.color_palette = [
            '#1f4e79', '#28a745', '#ffc107', '#fd7e14', '#dc3545',
            '#6f42c1', '#20c997', '#fd7e14', '#e83e8c', '#6c757d'
        ]
    
    def create_congestion_timeline(
        self,
        traffic_data: List[TrafficData],
        road_ids: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a timeline chart of congestion levels.
        
        Args:
            traffic_data: List of traffic data points
            road_ids: Specific road IDs to include (None for all)
            
        Returns:
            Plotly figure
        """
        if road_ids:
            filtered_data = [t for t in traffic_data if t.road_id in road_ids]
        else:
            filtered_data = traffic_data
        
        if not filtered_data:
            return go.Figure()
        
        df_data = []
        for traffic in filtered_data:
            df_data.append({
                'timestamp': traffic.timestamp,
                'road_id': traffic.road_id,
                'congestion_level': traffic.congestion_level.value,
                'congestion_name': traffic.congestion_level.name,
                'average_speed': traffic.average_speed,
                'traffic_volume': traffic.traffic_volume
            })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return go.Figure()
        
        fig = px.line(
            df,
            x='timestamp',
            y='congestion_level',
            color='road_id',
            title='Traffic Congestion Timeline',
            labels={
                'congestion_level': 'Congestion Level',
                'timestamp': 'Time'
            },
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            yaxis=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['Free Flow', 'Light', 'Moderate', 'Heavy', 'Severe']
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_speed_volume_correlation(
        self,
        traffic_data: List[TrafficData]
    ) -> go.Figure:
        """
        Create speed vs volume correlation chart.
        
        Args:
            traffic_data: List of traffic data points
            
        Returns:
            Plotly figure
        """
        if not traffic_data:
            return go.Figure()
        
        speeds = [t.average_speed for t in traffic_data]
        volumes = [t.traffic_volume for t in traffic_data]
        congestion_levels = [t.congestion_level.name for t in traffic_data]
        road_types = [t.road_type.value for t in traffic_data]
        
        fig = px.scatter(
            x=volumes,
            y=speeds,
            color=congestion_levels,
            symbol=road_types,
            title='Speed vs Traffic Volume Correlation',
            labels={
                'x': 'Traffic Volume (vehicles/hour)',
                'y': 'Average Speed (km/h)',
                'color': 'Congestion Level',
                'symbol': 'Road Type'
            },
            color_discrete_map={
                'FREE_FLOW': '#28a745',
                'LIGHT': '#ffc107',
                'MODERATE': '#fd7e14',
                'HEAVY': '#dc3545',
                'SEVERE': '#721c24'
            }
        )
        
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(volumes, speeds)
            line_x = [min(volumes), max(volumes)]
            line_y = [slope * x + intercept for x in line_x]
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'Trend (R²={r_value**2:.3f})',
                line=dict(color='red', dash='dash')
            ))
        except ImportError:
            pass  # Skip trend line if scipy not available
        
        return fig
    
    def create_hourly_patterns(
        self,
        traffic_data: List[TrafficData]
    ) -> go.Figure:
        """
        Create hourly traffic patterns chart.
        
        Args:
            traffic_data: List of traffic data points
            
        Returns:
            Plotly figure
        """
        if not traffic_data:
            return go.Figure()
        
        hourly_stats = {}
        for traffic in traffic_data:
            hour = traffic.timestamp.hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    'volumes': [],
                    'speeds': [],
                    'congestion': []
                }
            
            hourly_stats[hour]['volumes'].append(traffic.traffic_volume)
            hourly_stats[hour]['speeds'].append(traffic.average_speed)
            hourly_stats[hour]['congestion'].append(traffic.congestion_level.value)
        
        hours = sorted(hourly_stats.keys())
        avg_volumes = [np.mean(hourly_stats[h]['volumes']) for h in hours]
        avg_speeds = [np.mean(hourly_stats[h]['speeds']) for h in hours]
        avg_congestion = [np.mean(hourly_stats[h]['congestion']) for h in hours]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Average Traffic Volume', 'Average Speed', 'Average Congestion Level'],
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=avg_volumes,
                mode='lines+markers',
                name='Traffic Volume',
                line=dict(color='#1f4e79')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=avg_speeds,
                mode='lines+markers',
                name='Average Speed',
                line=dict(color='#28a745')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=avg_congestion,
                mode='lines+markers',
                name='Congestion Level',
                line=dict(color='#dc3545')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Hourly Traffic Patterns',
            height=800,
            showlegend=False
        )
        
        for i in range(1, 4):
            fig.update_xaxes(
                title_text='Hour of Day' if i == 3 else '',
                tickvals=list(range(0, 24, 2)),
                row=i, col=1
            )
        
        fig.update_yaxes(title_text='Vehicles/Hour', row=1, col=1)
        fig.update_yaxes(title_text='km/h', row=2, col=1)
        fig.update_yaxes(
            title_text='Level',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['Free', 'Light', 'Moderate', 'Heavy', 'Severe'],
            row=3, col=1
        )
        
        return fig
    
    def create_congestion_heatmap(
        self,
        traffic_data: List[TrafficData]
    ) -> go.Figure:
        """
        Create congestion heatmap by hour and day.
        
        Args:
            traffic_data: List of traffic data points
            
        Returns:
            Plotly figure
        """
        if not traffic_data:
            return go.Figure()
        
        heatmap_data = {}
        for traffic in traffic_data:
            hour = traffic.timestamp.hour
            day = traffic.timestamp.strftime('%A')
            
            if day not in heatmap_data:
                heatmap_data[day] = {}
            if hour not in heatmap_data[day]:
                heatmap_data[day][hour] = []
            
            heatmap_data[day][hour].append(traffic.congestion_level.value)
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = list(range(24))
        
        z_data = []
        for day in days:
            row = []
            for hour in hours:
                if day in heatmap_data and hour in heatmap_data[day]:
                    avg_congestion = np.mean(heatmap_data[day][hour])
                else:
                    avg_congestion = 0
                row.append(avg_congestion)
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=hours,
            y=days,
            colorscale='RdYlGn_r',
            colorbar=dict(
                title='Congestion Level',
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['Free Flow', 'Light', 'Moderate', 'Heavy', 'Severe']
            ),
            hovertemplate='Hour: %{x}<br>Day: %{y}<br>Congestion: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Traffic Congestion Heatmap by Hour and Day',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )
        
        return fig