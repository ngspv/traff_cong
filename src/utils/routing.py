"""
Route optimization and alternative path suggestion algorithm.
Uses predicted congestion patterns to recommend optimal routes.
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from geopy.distance import geodesic
import heapq
import logging
from pathlib import Path
import pickle

from ..data.schema import Location, CongestionLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Road:
    """Represents a road segment in the transportation network."""
    road_id: str
    start_location: Location
    end_location: Location
    length_km: float
    road_type: str
    speed_limit: float
    capacity: int
    current_congestion: CongestionLevel = CongestionLevel.FREE_FLOW
    predicted_congestion: Optional[CongestionLevel] = None
    predicted_travel_time: Optional[float] = None
    
    def calculate_travel_time(self, congestion_level: CongestionLevel = None) -> float:
        """Calculate travel time based on congestion level."""
        if congestion_level is None:
            congestion_level = self.current_congestion
        
        speed_factors = {
            CongestionLevel.FREE_FLOW: 1.0,
            CongestionLevel.LIGHT: 0.8,
            CongestionLevel.MODERATE: 0.6,
            CongestionLevel.HEAVY: 0.4,
            CongestionLevel.SEVERE: 0.2
        }
        
        effective_speed = self.speed_limit * speed_factors[congestion_level]
        travel_time_hours = self.length_km / max(effective_speed, 1.0)  # Prevent division by zero
        return travel_time_hours * 60  # Convert to minutes
    
    def get_cost(self, weight_time: float = 0.7, weight_distance: float = 0.3) -> float:
        """Calculate route cost considering time and distance."""
        travel_time = self.calculate_travel_time(self.predicted_congestion or self.current_congestion)
        return weight_time * travel_time + weight_distance * self.length_km


@dataclass
class Route:
    """Represents a complete route between two points."""
    route_id: str
    roads: List[Road]
    start_location: Location
    end_location: Location
    total_distance: float
    total_time: float
    total_cost: float
    congestion_score: float
    alternative_rank: int = 0
    
    @classmethod
    def from_roads(
        cls,
        route_id: str,
        roads: List[Road],
        weight_time: float = 0.7,
        weight_distance: float = 0.3
    ) -> 'Route':
        """Create route from list of roads."""
        if not roads:
            raise ValueError("Route must contain at least one road")
        
        start_location = roads[0].start_location
        end_location = roads[-1].end_location
        total_distance = sum(road.length_km for road in roads)
        total_time = sum(road.calculate_travel_time(
            road.predicted_congestion or road.current_congestion
        ) for road in roads)
        total_cost = sum(road.get_cost(weight_time, weight_distance) for road in roads)
        
        congestion_scores = [road.current_congestion.value for road in roads]
        congestion_score = np.mean(congestion_scores)
        
        return cls(
            route_id=route_id,
            roads=roads,
            start_location=start_location,
            end_location=end_location,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            congestion_score=congestion_score
        )


class RoadNetwork:
    """Represents the road network as a graph."""
    
    def __init__(self):
        """Initialize the road network."""
        self.graph = nx.DiGraph()
        self.roads = {}  # road_id -> Road
        self.intersections = {}  # intersection_id -> Location
        
    def add_road(self, road: Road) -> None:
        """Add a road to the network."""
        self.roads[road.road_id] = road
        
        start_node = self._location_to_node_id(road.start_location)
        end_node = self._location_to_node_id(road.end_location)
        
        if start_node not in self.graph:
            self.graph.add_node(start_node, location=road.start_location)
            self.intersections[start_node] = road.start_location
            
        if end_node not in self.graph:
            self.graph.add_node(end_node, location=road.end_location)
            self.intersections[end_node] = road.end_location
        
        self.graph.add_edge(
            start_node,
            end_node,
            road_id=road.road_id,
            weight=road.get_cost(),
            travel_time=road.calculate_travel_time(),
            distance=road.length_km,
            congestion=road.current_congestion.value
        )
    
    def _location_to_node_id(self, location: Location) -> str:
        """Convert location to node ID."""
        return f"{location.latitude:.6f},{location.longitude:.6f}"
    
    def _node_id_to_location(self, node_id: str) -> Location:
        """Convert node ID back to location."""
        lat_str, lon_str = node_id.split(',')
        return Location(latitude=float(lat_str), longitude=float(lon_str))
    
    def find_nearest_node(self, location: Location) -> str:
        """Find the nearest node to a given location."""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_location in self.intersections.items():
            distance = geodesic(
                (location.latitude, location.longitude),
                (node_location.latitude, node_location.longitude)
            ).kilometers
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def update_road_congestion(self, road_id: str, congestion: CongestionLevel) -> None:
        """Update congestion level for a specific road."""
        if road_id in self.roads:
            self.roads[road_id].current_congestion = congestion
            
            road = self.roads[road_id]
            start_node = self._location_to_node_id(road.start_location)
            end_node = self._location_to_node_id(road.end_location)
            
            if self.graph.has_edge(start_node, end_node):
                self.graph[start_node][end_node]['weight'] = road.get_cost()
                self.graph[start_node][end_node]['travel_time'] = road.calculate_travel_time()
                self.graph[start_node][end_node]['congestion'] = congestion.value
    
    def update_predicted_congestion(self, predictions: Dict[str, CongestionLevel]) -> None:
        """Update predicted congestion for multiple roads."""
        for road_id, predicted_congestion in predictions.items():
            if road_id in self.roads:
                self.roads[road_id].predicted_congestion = predicted_congestion
                
                road = self.roads[road_id]
                start_node = self._location_to_node_id(road.start_location)
                end_node = self._location_to_node_id(road.end_location)
                
                if self.graph.has_edge(start_node, end_node):
                    predicted_travel_time = road.calculate_travel_time(predicted_congestion)
                    self.graph[start_node][end_node]['predicted_travel_time'] = predicted_travel_time
                    self.graph[start_node][end_node]['predicted_congestion'] = predicted_congestion.value
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            'num_roads': len(self.roads),
            'num_intersections': len(self.intersections),
            'num_edges': self.graph.number_of_edges(),
            'num_nodes': self.graph.number_of_nodes(),
            'avg_congestion': np.mean([road.current_congestion.value for road in self.roads.values()]),
            'congestion_distribution': {
                level.name: sum(1 for road in self.roads.values() 
                               if road.current_congestion == level)
                for level in CongestionLevel
            }
        }


class RouteOptimizer:
    """Optimizes routes based on various criteria."""
    
    def __init__(self, road_network: RoadNetwork):
        """Initialize route optimizer."""
        self.network = road_network
        
    def find_shortest_path(
        self,
        start_location: Location,
        end_location: Location,
        weight: str = 'weight'
    ) -> Optional[Route]:
        """Find shortest path between two locations."""
        start_node = self.network.find_nearest_node(start_location)
        end_node = self.network.find_nearest_node(end_location)
        
        if not start_node or not end_node:
            logger.warning("Could not find nearest nodes for start/end locations")
            return None
        
        try:
            path = nx.shortest_path(
                self.network.graph,
                start_node,
                end_node,
                weight=weight
            )
            
            return self._path_to_route(path, "shortest_path")
            
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_location} and {end_location}")
            return None
    
    def find_fastest_path(
        self,
        start_location: Location,
        end_location: Location,
        use_predictions: bool = True
    ) -> Optional[Route]:
        """Find fastest path considering current or predicted congestion."""
        weight_key = 'predicted_travel_time' if use_predictions else 'travel_time'
        
        temp_graph = self.network.graph.copy()
        
        for u, v, data in temp_graph.edges(data=True):
            if weight_key in data:
                temp_graph[u][v]['temp_weight'] = data[weight_key]
            else:
                temp_graph[u][v]['temp_weight'] = data.get('travel_time', data.get('weight', float('inf')))
        
        start_node = self.network.find_nearest_node(start_location)
        end_node = self.network.find_nearest_node(end_location)
        
        if not start_node or not end_node:
            return None
        
        try:
            path = nx.shortest_path(
                temp_graph,
                start_node,
                end_node,
                weight='temp_weight'
            )
            
            return self._path_to_route(path, "fastest_path")
            
        except nx.NetworkXNoPath:
            return None
    
    def find_alternative_routes(
        self,
        start_location: Location,
        end_location: Location,
        num_alternatives: int = 3,
        max_detour_factor: float = 1.5
    ) -> List[Route]:
        """Find multiple alternative routes."""
        routes = []
        
        primary_route = self.find_fastest_path(start_location, end_location)
        if primary_route:
            primary_route.alternative_rank = 1
            routes.append(primary_route)
        
        start_node = self.network.find_nearest_node(start_location)
        end_node = self.network.find_nearest_node(end_location)
        
        if not start_node or not end_node:
            return routes
        
        temp_graph = self.network.graph.copy()
        
        for i in range(num_alternatives - 1):
            try:
                if routes:
                    last_route = routes[-1]
                    path_edges = self._route_to_path_edges(last_route)
                    
                    removed_edges = []
                    for u, v in path_edges[::2]:  # Remove every other edge
                        if temp_graph.has_edge(u, v):
                            edge_data = temp_graph[u][v]
                            temp_graph.remove_edge(u, v)
                            removed_edges.append((u, v, edge_data))
                
                path = nx.shortest_path(
                    temp_graph,
                    start_node,
                    end_node,
                    weight='weight'
                )
                
                alternative_route = self._path_to_route(path, f"alternative_{i+2}")
                
                if primary_route and alternative_route.total_cost <= primary_route.total_cost * max_detour_factor:
                    alternative_route.alternative_rank = i + 2
                    routes.append(alternative_route)
                
                for u, v, data in removed_edges:
                    temp_graph.add_edge(u, v, **data)
                    
            except (nx.NetworkXNoPath, IndexError):
                break
        
        routes.sort(key=lambda r: r.total_cost)
        
        for i, route in enumerate(routes):
            route.alternative_rank = i + 1
        
        logger.info(f"Found {len(routes)} alternative routes")
        return routes
    
    def _path_to_route(self, path: List[str], route_id: str) -> Route:
        """Convert node path to Route object."""
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")
        
        roads = []
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            
            if self.network.graph.has_edge(start_node, end_node):
                edge_data = self.network.graph[start_node][end_node]
                road_id = edge_data['road_id']
                road = self.network.roads[road_id]
                roads.append(road)
        
        if not roads:
            raise ValueError("No roads found for path")
        
        return Route.from_roads(route_id, roads)
    
    def _route_to_path_edges(self, route: Route) -> List[Tuple[str, str]]:
        """Convert route to list of edge tuples."""
        edges = []
        for road in route.roads:
            start_node = self.network._location_to_node_id(road.start_location)
            end_node = self.network._location_to_node_id(road.end_location)
            edges.append((start_node, end_node))
        return edges


class TrafficAwareRouter:
    """Main router that integrates traffic predictions with route optimization."""
    
    def __init__(self, road_network: RoadNetwork, traffic_predictor=None):
        """Initialize traffic-aware router."""
        self.network = road_network
        self.optimizer = RouteOptimizer(road_network)
        self.traffic_predictor = traffic_predictor
        
    def set_traffic_predictor(self, predictor) -> None:
        """Set the traffic prediction model."""
        self.traffic_predictor = predictor
    
    def predict_traffic_for_routes(
        self,
        routes: List[Route],
        prediction_time: datetime,
        horizon_hours: int = 1
    ) -> Dict[str, List[CongestionLevel]]:
        """Predict traffic congestion for routes at specific time."""
        if self.traffic_predictor is None:
            logger.warning("No traffic predictor available")
            return {}
        
        predictions = {}
        
        for route in routes:
            route_predictions = []
            for road in route.roads:
                try:
                    predicted_level = self._predict_road_congestion(
                        road,
                        prediction_time,
                        horizon_hours
                    )
                    route_predictions.append(predicted_level)
                except Exception as e:
                    logger.warning(f"Prediction failed for road {road.road_id}: {e}")
                    route_predictions.append(road.current_congestion)
            
            predictions[route.route_id] = route_predictions
        
        return predictions
    
    def _predict_road_congestion(
        self,
        road: Road,
        prediction_time: datetime,
        horizon_hours: int
    ) -> CongestionLevel:
        """Predict congestion for a single road (placeholder implementation)."""
        hour = prediction_time.hour
        day_of_week = prediction_time.weekday()
        
        if (7 <= hour <= 9) or (17 <= hour <= 19):  # Rush hours
            if day_of_week < 5:  # Weekday
                base_congestion = CongestionLevel.HEAVY
            else:
                base_congestion = CongestionLevel.MODERATE
        elif day_of_week >= 5:  # Weekend
            base_congestion = CongestionLevel.LIGHT
        else:
            base_congestion = CongestionLevel.MODERATE
        
        if road.road_type == 'highway':
            adjustment = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        else:
            adjustment = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        
        final_level = max(0, min(4, base_congestion.value + adjustment))
        return CongestionLevel(final_level)
    
    def get_optimal_routes(
        self,
        start_location: Location,
        end_location: Location,
        departure_time: datetime,
        num_alternatives: int = 3,
        preferences: Dict[str, float] = None
    ) -> List[Route]:
        """Get optimal routes considering traffic predictions."""
        
        if preferences is None:
            preferences = {
                'weight_time': 0.6,
                'weight_distance': 0.2,
                'weight_comfort': 0.2  # Based on congestion level
            }
        
        routes = self.optimizer.find_alternative_routes(
            start_location,
            end_location,
            num_alternatives
        )
        
        if not routes:
            logger.warning("No routes found")
            return []
        
        traffic_predictions = self.predict_traffic_for_routes(
            routes,
            departure_time,
            horizon_hours=2
        )
        
        for route in routes:
            if route.route_id in traffic_predictions:
                predictions = traffic_predictions[route.route_id]
                
                for road, predicted_congestion in zip(route.roads, predictions):
                    road.predicted_congestion = predicted_congestion
                
                route.total_time = sum(
                    road.calculate_travel_time(road.predicted_congestion)
                    for road in route.roads
                )
                
                comfort_score = 5 - np.mean([road.predicted_congestion.value for road in route.roads])
                
                route.total_cost = (
                    preferences['weight_time'] * route.total_time +
                    preferences['weight_distance'] * route.total_distance +
                    preferences['weight_comfort'] * (5 - comfort_score)
                )
        
        routes.sort(key=lambda r: r.total_cost)
        
        for i, route in enumerate(routes):
            route.alternative_rank = i + 1
        
        logger.info(f"Optimized {len(routes)} routes with traffic predictions")
        return routes
    
    def save_network(self, file_path: str) -> None:
        """Save road network to file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.network, f)
        logger.info(f"Road network saved to {file_path}")
    
    def load_network(self, file_path: str) -> None:
        """Load road network from file."""
        with open(file_path, 'rb') as f:
            self.network = pickle.load(f)
        self.optimizer = RouteOptimizer(self.network)
        logger.info(f"Road network loaded from {file_path}")


def create_sample_network() -> RoadNetwork:
    """Create a sample road network for testing."""
    network = RoadNetwork()
    
    locations = {
        'A': Location(40.7589, -73.9851),  # Times Square
        'B': Location(40.7505, -73.9934),  # Herald Square  
        'C': Location(40.7614, -73.9776),  # Central Park South
        'D': Location(40.7831, -73.9712),  # Upper West Side
        'E': Location(40.7282, -73.7949),  # Queens
        'F': Location(40.6892, -74.0445),  # Brooklyn
    }
    
    roads_data = [
        ('road_1', 'A', 'B', 1.2, 'arterial', 50, 2000),
        ('road_2', 'B', 'C', 1.5, 'arterial', 50, 2000),
        ('road_3', 'C', 'D', 2.1, 'arterial', 60, 2500),
        ('road_4', 'A', 'E', 15.2, 'highway', 80, 4000),
        ('road_5', 'B', 'F', 8.7, 'highway', 70, 3500),
        ('road_6', 'A', 'C', 2.0, 'collector', 45, 1500),
        ('road_7', 'D', 'E', 12.3, 'highway', 75, 3800),
        ('road_8', 'C', 'F', 9.1, 'arterial', 55, 2200),
    ]
    
    for road_id, start, end, length, road_type, speed_limit, capacity in roads_data:
        road = Road(
            road_id=road_id,
            start_location=locations[start],
            end_location=locations[end],
            length_km=length,
            road_type=road_type,
            speed_limit=speed_limit,
            capacity=capacity,
            current_congestion=np.random.choice(list(CongestionLevel))
        )
        network.add_road(road)
    
    logger.info(f"Created sample network with {len(roads_data)} roads")
    return network