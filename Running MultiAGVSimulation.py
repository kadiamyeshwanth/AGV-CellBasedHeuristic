import pygame
import sys
import math
import networkx as nx
from collections import defaultdict, deque
import heapq
import random
from pygame.locals import *

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 1200, 900
GRID_SIZE = 15
CELL_SIZE = 50
MARGIN = 50
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-AGV Path Optimization Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 80, 255)
YELLOW = (255, 255, 80)
GRAY = (220, 220, 220)
PURPLE = (160, 80, 160)
ORANGE = (255, 165, 80)
CYAN = (80, 255, 255)
DARK_GREEN = (0, 120, 0)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
SHADOW_COLOR = (150, 150, 150)
BACKGROUND_START = (180, 200, 220)  # Gradient start (light blue-gray)
BACKGROUND_END = (220, 230, 240)    # Gradient end (lighter)

# Fonts (fallback to Arial if Roboto is unavailable)
font = pygame.font.SysFont('Roboto', 16, bold=True) or pygame.font.SysFont('Arial', 16, bold=True)
large_font = pygame.font.SysFont('Roboto', 24, bold=True) or pygame.font.SysFont('Arial', 24, bold=True)
title_font = pygame.font.SysFont('Roboto', 28, bold=True) or pygame.font.SysFont('Arial', 28, bold=True)

class AGV:
    def __init__(self, id, start_pos, targets, color):
        self.id = id
        self.pos = start_pos
        self.targets = deque(targets)
        self.path = []
        self.color = color
        self.speed = 0.08
        self.waiting = 0
        self.delivered = False
        self.path_history = deque([start_pos], maxlen=100)
        self.collision_count = 0
        self.total_distance = 0
        self.G = None
        self.congestion_data = None
        self.cell_map = None
        self.flow_data = None  # For dynamic multicommodity flow

    def update_position(self, occupied_cells):
        if self.delivered or not self.targets:
            return False

        if not self.path and self.targets:
            next_target = self.targets[0]
            try:
                self.path = self.find_path(self.pos, next_target) or []
                if not self.path:
                    # If no path found, try with rounded position
                    rounded_pos = (round(self.pos[0]), round(self.pos[1]))
                    self.path = self.find_path(rounded_pos, next_target) or []
                    if not self.path:
                        return False
            except Exception as e:
                # Handle any errors in path finding
                print(f"Path finding error: {e}")
                return False

        # Safety check for empty path
        if not self.path:
            return False

        next_pos = self.path[0]

        # Check if the next position is occupied by another AGV
        rounded_next_pos = (round(next_pos[0]), round(next_pos[1]))
        if rounded_next_pos in occupied_cells and occupied_cells[rounded_next_pos] != self.id:
            self.waiting += 1
            if self.waiting > 25:
                self.waiting = 0
                return "replan"
            return True

        self.waiting = 0

        dx = next_pos[0] - self.pos[0]
        dy = next_pos[1] - self.pos[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance < self.speed:
            self.pos = next_pos
            self.path_history.append(self.pos)
            if self.path:
                self.path.pop(0)
                self.total_distance += 1
            if not self.path and self.targets:
                self.targets.popleft()
                if not self.targets:
                    self.delivered = True
            return True
        else:
            # Update position with floating point values
            new_x = self.pos[0] + (dx/distance) * self.speed
            new_y = self.pos[1] + (dy/distance) * self.speed
            self.pos = (new_x, new_y)
            return True

    def find_path(self, start, end, algorithm='astar'):
        if start == end:
            return []

        if algorithm == 'astar':
            return self.astar_path(start, end)
        elif algorithm == 'aco':
            return self.aco_path(start, end)
        elif algorithm == 'sslph':
            return self.sslph_path(start, end)
        elif algorithm == 'cell_based':
            return self.cell_based_path(start, end)
        elif algorithm == 'dynamic_mcf':
            return self.dynamic_multicommodity_flow_path(start, end)
        elif algorithm == 'flow_concentrated':
            return self.flow_concentrated_cells_path(start, end)

    def astar_path(self, start, end):
        # Ensure start and end points are in the graph
        if self.G is None:
            return None
            
        # Round start and end positions to ensure they're valid nodes in the graph
        if isinstance(start, tuple) and len(start) == 2:
            start = (round(start[0]), round(start[1]))
        if isinstance(end, tuple) and len(end) == 2:
            end = (round(end[0]), round(end[1]))
            
        # Check if start and end are valid nodes
        if start not in self.G:
            closest = min(self.G.nodes(), key=lambda n: self.heuristic(n, start), default=None)
            if closest:
                start = closest
            else:
                return None
                
        if end not in self.G:
            closest = min(self.G.nodes(), key=lambda n: self.heuristic(n, end), default=None)
            if closest:
                end = closest
            else:
                return None
        
        # Standard A* implementation
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic(start, end)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path[1:] if len(path) > 1 else path

            # Get neighbors safely
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                congestion = self.get_congestion(current, neighbor) / 10
                tentative_g = g_score[current] + 1 + congestion

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def aco_path(self, start, end, iterations=50, ants=10, alpha=1, beta=2, evaporation=0.5):
        pheromone = defaultdict(lambda: 1.0)
        best_path = None
        best_distance = float('inf')

        for _ in range(iterations):
            paths = []
            distances = []

            for _ in range(ants):
                current = start
                path = [current]
                visited = {current}
                distance = 0

                while current != end:
                    neighbors = [n for n in self.get_neighbors(current) if n not in visited]
                    if not neighbors:
                        break

                    probabilities = []
                    total = 0
                    for n in neighbors:
                        p = (pheromone[(current, n)] ** alpha) * ((1/self.heuristic(n, end)) ** beta)
                        total += p
                        probabilities.append((n, p))

                    if total == 0:
                        break

                    probabilities = [(n, p/total) for n, p in probabilities]
                    next_node = random.choices([n for n, _ in probabilities], 
                                             [p for _, p in probabilities])[0]
                    
                    path.append(next_node)
                    distance += self.heuristic(current, next_node)
                    visited.add(next_node)
                    current = next_node

                if current == end:
                    paths.append(path)
                    distances.append(distance)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path

            for edge in pheromone:
                pheromone[edge] *= (1 - evaporation)

            for path, distance in zip(paths, distances):
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    pheromone[edge] += 1.0 / distance

        return best_path[1:] if best_path else None

    def sslph_path(self, start, end, population_size=20, iterations=30):
        def generate_initial_path():
            path = [start]
            current = start
            while current != end:
                neighbors = [n for n in self.get_neighbors(current) if n not in path]
                if not neighbors:
                    return None
                next_node = min(neighbors, key=lambda n: self.heuristic(n, end))
                path.append(next_node)
                current = next_node
            return path

        population = []
        for _ in range(population_size):
            path = generate_initial_path()
            if path:
                population.append(path)

        for _ in range(iterations):
            population = sorted(population, key=lambda p: sum(self.heuristic(p[i], p[i+1]) for i in range(len(p)-1)))[:population_size//2]
            new_population = population.copy()
            for _ in range(population_size - len(population)):
                p1, p2 = random.sample(population, 2)
                crossover_point = random.randint(1, min(len(p1), len(p2))-1)
                new_path = p1[:crossover_point] + [n for n in p2[crossover_point:] if n not in p1[:crossover_point]]
                if new_path[-1] != end:
                    new_path = generate_initial_path()
                if new_path:
                    new_population.append(new_path)
            population = new_population

        best_path = min(population, key=lambda p: sum(self.heuristic(p[i], p[i+1]) for i in range(len(p)-1)))
        return best_path[1:] if best_path else None

    def cell_based_path(self, start, end, k=2):
        """
        Enhanced Cell-Based Local Search Heuristics
        Divides the grid into cells and performs local search within and between cells
        """
        if not self.cell_map:
            return self.astar_path(start, end)

        def get_cell(pos):
            return self.cell_map.get(pos, 0)

        # Get the cells for start and end positions
        start_cell = get_cell(start)
        end_cell = get_cell(end)
        
        # First phase: Find best route between cells
        cell_graph = defaultdict(list)
        for node in self.G.nodes():
            cell = get_cell(node)
            cell_graph[cell].append(node)
        
        # Create connections between adjacent cells
        cell_connections = defaultdict(list)
        for (u, v) in self.G.edges():
            cell_u = get_cell(u)
            cell_v = get_cell(v)
            if cell_u != cell_v:
                if cell_v not in cell_connections[cell_u]:
                    cell_connections[cell_u].append(cell_v)
                if cell_u not in cell_connections[cell_v]:
                    cell_connections[cell_v].append(cell_u)
        
        # Find path between cells
        cell_path = self.find_cell_path(start_cell, end_cell, cell_connections)
        if not cell_path:
            return self.astar_path(start, end)
        
        # Second phase: Find optimal path through each cell
        complete_path = [start]
        current = start
        
        for i in range(1, len(cell_path)):
            # Find a node in the current cell that connects to the next cell
            current_cell = cell_path[i-1]
            next_cell = cell_path[i]
            
            # Find nodes in current cell that connect to nodes in next cell
            border_nodes = []
            for u in cell_graph[current_cell]:
                for v in cell_graph[next_cell]:
                    if self.G.has_edge(u, v):
                        border_nodes.append((u, v))
            
            if not border_nodes:
                # If no direct connection, find best path within current cell to any node
                # that may have a path to the next cell
                best_node = None
                best_score = float('inf')
                for node in cell_graph[current_cell]:
                    for next_node in cell_graph[next_cell]:
                        try:
                            path = nx.shortest_path(self.G, node, next_node)
                            score = len(path) + self.heuristic(current, node)
                            if score < best_score:
                                best_score = score
                                best_node = node
                        except nx.NetworkXNoPath:
                            continue
                
                if best_node:
                    # Find path from current position to best node
                    inner_path = self.astar_path(current, best_node)
                    if inner_path:
                        complete_path.extend(inner_path)
                        current = best_node
            else:
                # Use the best border node based on distance and congestion
                best_border = min(border_nodes, 
                                 key=lambda x: self.heuristic(current, x[0]) + 
                                              self.get_congestion(x[0], x[1]))
                
                inner_path = self.astar_path(current, best_border[0])
                if inner_path:
                    complete_path.extend(inner_path)
                    complete_path.append(best_border[1])
                    current = best_border[1]
        
        # Final phase: Path from last node to target
        final_path = self.astar_path(current, end)
        if final_path:
            complete_path.extend(final_path)
        
        return complete_path[1:] if len(complete_path) > 1 else None

    def find_cell_path(self, start_cell, end_cell, cell_connections):
        """
        Finds a path between cells using A* search
        """
        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_cell] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start_cell] = abs(start_cell % 3 - end_cell % 3) + abs(start_cell // 3 - end_cell // 3)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in cell_connections[current]:
                tentative_g = g_score[current] + 1

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = abs(neighbor % 3 - end_cell % 3) + abs(neighbor // 3 - end_cell // 3)
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def dynamic_multicommodity_flow_path(self, start, end):
        """
        Dynamic Multicommodity Flow path finding
        Takes into account flow of multiple AGVs and dynamically updates path based on demand changes
        """
        if not self.flow_data:
            return self.astar_path(start, end)
        
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic(start, end)
        
        # Current time step (used to model dynamic demands)
        time_step = 0
        
        # Flow capacity of each edge
        capacities = defaultdict(lambda: 3.0)  # Default capacity for each edge
        
        # Current flow for each edge
        flow = self.flow_data
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            time_step += 1  # Increment time step for dynamic adjustments
            
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path[1:]
            
            for neighbor in self.get_neighbors(current):
                edge = (current, neighbor)
                
                # Get dynamic edge cost based on congestion and time step
                current_flow = flow.get(edge, 0)
                
                # Calculate residual capacity
                residual = max(0.1, capacities[edge] - current_flow)
                
                # Dynamic cost incorporates congestion, time, and residual capacity
                edge_cost = 1.0 + (current_flow / max(0.1, capacities[edge])) * 2.0
                
                # Temporal factor: cost increases with time if flow is high
                temporal_factor = 1.0 + 0.01 * time_step * (current_flow / max(0.1, capacities[edge]))
                
                # Demand prediction - assume peak flows occur at certain time steps
                if time_step % 20 < 5 and current_flow > 0:  # Peak periods
                    peak_factor = 1.5
                else:
                    peak_factor = 1.0
                
                total_cost = edge_cost * temporal_factor * peak_factor
                
                tentative_g = g_score[current] + total_cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If no path found, try regular A*
        return self.astar_path(start, end)

    def flow_concentrated_cells_path(self, start, end):
        """
        Uses identified high-traffic (flow-concentrated) cells to guide path finding
        """
        if not self.cell_map or not self.flow_data:
            return self.astar_path(start, end)
        
        # Identify flow-concentrated cells
        cell_flow = defaultdict(float)
        for (u, v), flow_value in self.flow_data.items():
            if flow_value > 0:
                cell_u = self.cell_map.get(u, 0)
                cell_v = self.cell_map.get(v, 0)
                cell_flow[cell_u] += flow_value
                cell_flow[cell_v] += flow_value
        
        # Determine high-traffic cells (top 30% by flow)
        high_traffic_threshold = sorted(cell_flow.values(), reverse=True)
        if len(high_traffic_threshold) > 3:
            high_traffic_threshold = high_traffic_threshold[int(len(high_traffic_threshold) * 0.3)]
        else:
            high_traffic_threshold = 0
        
        high_traffic_cells = {cell for cell, flow in cell_flow.items() if flow >= high_traffic_threshold}
        
        # A* search that tries to avoid high-traffic cells when possible
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic(start, end)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path[1:]

            for neighbor in self.get_neighbors(current):
                # Base cost is distance
                base_cost = 1.0
                
                # Add penalty for high traffic cells
                neighbor_cell = self.cell_map.get(neighbor, 0)
                traffic_penalty = 2.0 if neighbor_cell in high_traffic_cells else 0.0
                
                # Get edge-specific congestion
                congestion = self.get_congestion(current, neighbor) / 10
                
                # Total edge cost
                edge_cost = base_cost + traffic_penalty + congestion
                
                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return self.astar_path(start, end)

    def get_neighbors(self, node):
        if self.G is None:
            return []
        
        # Handle floating point coordinates by finding the closest node
        if isinstance(node, tuple) and len(node) == 2:
            x, y = node
            # Check if node exists exactly in graph
            if node in self.G:
                return list(self.G.neighbors(node))
            
            # If node has floating point values, round to nearest integer node
            if isinstance(x, float) or isinstance(y, float):
                rounded_node = (round(x), round(y))
                if rounded_node in self.G:
                    return list(self.G.neighbors(rounded_node))
                
                # If still not found, try to find the closest node in the graph
                min_dist = float('inf')
                closest_node = None
                for graph_node in self.G.nodes():
                    dist = self.heuristic(node, graph_node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = graph_node
                
                if closest_node and min_dist < 1.0:  # Only use if reasonably close
                    return list(self.G.neighbors(closest_node))
        
        # If all else fails, return empty list to avoid errors
        return []

    def get_congestion(self, u, v):
        if self.congestion_data is None:
            return 0
        
        # Handle floating point coordinates
        if isinstance(u, tuple) and isinstance(v, tuple) and len(u) == 2 and len(v) == 2:
            # Try exact coordinates first
            if (u, v) in self.congestion_data:
                return self.congestion_data.get((u, v), 0)
            
            # Try rounded coordinates
            rounded_u = (round(u[0]), round(u[1]))
            rounded_v = (round(v[0]), round(v[1]))
            return self.congestion_data.get((rounded_u, rounded_v), 0)
        
        return 0

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AGVSimulator:
    def __init__(self):
        self.last_configuration = None  # Store last configuration for replay
        self.reset()

    def store_current_configuration(self):
        """Store the current AGV configuration for replay"""
        self.last_configuration = {
            'agvs': [(agv.pos, list(agv.targets), agv.color) for agv in self.agvs]
        }

    def replay_last_configuration(self):
        """Replay the last stored configuration"""
        if not self.last_configuration:
            self.message = "No previous configuration to replay"
            return
            
        self.reset()
        for pos, targets, color in self.last_configuration['agvs']:
            self.add_agv(pos, targets)
            # Set the color to match the original
            if self.agvs:
                self.agvs[-1].color = color
        self.message = "Replaying last configuration"

    def create_grid_graph(self):
        G = nx.Graph()
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                G.add_node((x, y))
                if x > 0:
                    G.add_edge((x, y), (x-1, y))
                if y > 0:
                    G.add_edge((x, y), (x, y-1))
        return G

    def initialize_congestion_data(self):
        self.congestion_data = defaultdict(int)
        for u, v in self.G.edges():
            self.congestion_data[(u, v)] = 0
            self.congestion_data[(v, u)] = 0

    def initialize_flow_data(self):
        self.flow_data = defaultdict(int)
        for u, v in self.G.edges():
            self.flow_data[(u, v)] = 0
            self.flow_data[(v, u)] = 0

    def create_cell_map(self):
        cell_map = {}
        cell_size = GRID_SIZE // 3
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                cell_x = x // cell_size
                cell_y = y // cell_size
                cell_map[(x, y)] = cell_x + cell_y * 3
        return cell_map

    def optimize_guide_path(self):
        """
        Guide Path Design - Creates optimal paths for AGVs
        Uses MST and shortest paths to reduce complexity while ensuring connectivity
        """
        if not self.G.edges():
            return

        # Create a new graph for the optimized guide path
        optimized_G = nx.Graph()
        
        # Add all nodes
        for node in self.G.nodes():
            optimized_G.add_node(node)
        
        # Add edges from MST for baseline connectivity
        mst = nx.minimum_spanning_tree(self.G)
        for u, v in mst.edges():
            optimized_G.add_edge(u, v)
            self.congestion_data[(u, v)] = self.congestion_data.get((u, v), 0)
            self.congestion_data[(v, u)] = self.congestion_data.get((v, u), 0)
        
        # Add additional optimal paths between AGV sources and destinations
        for agv in self.agvs:
            # Start position to all targets
            source = (round(agv.pos[0]), round(agv.pos[1]))
            for target in agv.targets:
                if source not in self.G.nodes() or target not in self.G.nodes():
                    self.message = f"Invalid node: AGV {agv.id} at {source} or target {target}"
                    continue
                
                try:
                    # Find shortest path in original graph
                    path = nx.shortest_path(self.G, source, target)
                    
                    # Add path edges to optimized graph
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        optimized_G.add_edge(u, v)
                        self.congestion_data[(u, v)] = self.congestion_data.get((u, v), 0)
                        self.congestion_data[(v, u)] = self.congestion_data.get((v, u), 0)
                except nx.NetworkXNoPath:
                    self.message = f"No path from {source} to {target} for AGV {agv.id}"
                    continue
        
        # Analyze high-traffic areas to add redundant paths for these areas
        congested_edges = sorted(self.congestion_data.items(), key=lambda x: x[1], reverse=True)
        for (u, v), congestion in congested_edges[:10]:  # Add redundancy for top 10 congested edges
            if congestion > 0 and (u, v) in self.G.edges():
                # Find alternative paths around congested edges
                try:
                    temp_G = self.G.copy()
                    temp_G.remove_edge(u, v)
                    alt_path = nx.shortest_path(temp_G, u, v)
                    
                    # Add alternative path to optimized graph
                    for i in range(len(alt_path)-1):
                        optimized_G.add_edge(alt_path[i], alt_path[i+1])
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue

        # Update simulator graph
        self.G = optimized_G
        self.message = "Guide path optimized"

    def eliminate_subtours(self):
        """
        Basic elimination of unconnected components
        """
        if not self.G.edges():
            return

        components = list(nx.connected_components(self.G))
        if len(components) > 1:
            for i in range(len(components)-1):
                c1, c2 = components[i], components[i+1]
                min_dist = float('inf')
                best_edge = None
                for n1 in c1:
                    for n2 in c2:
                        dist = self.heuristic(n1, n2)
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (n1, n2)
                if best_edge:
                    self.G.add_edge(*best_edge)
                    self.congestion_data[best_edge] = 0
                    self.congestion_data[(best_edge[1], best_edge[0])] = 0
            self.message = "Subtours eliminated"

    def redundant_arc_elimination(self):
        """
        Identifies and removes unnecessary edges that don't affect connectivity or critical paths
        """
        if not self.G.edges():
            return
        
        # Make a copy of the graph to work with
        G_copy = self.G.copy()
        
        # Get critical paths for all AGVs
        critical_edges = set()
        for agv in self.agvs:
            if not agv.delivered and agv.targets:
                source = (round(agv.pos[0]), round(agv.pos[1]))
                for target in agv.targets:
                    try:
                        path = nx.shortest_path(self.G, source, target)
                        for i in range(len(path)-1):
                            critical_edges.add((path[i], path[i+1]))
                            critical_edges.add((path[i+1], path[i]))
                    except nx.NetworkXNoPath:
                        continue
        
        # Calculate edge betweenness centrality to find critical edges
        edge_betweenness = nx.edge_betweenness_centrality(self.G)
        
        # Sort edges by betweenness (less important edges first)
        edges_by_importance = sorted(edge_betweenness.items(), key=lambda x: x[1])
        
        # Remove redundant edges
        removed_count = 0
        for edge, _ in edges_by_importance:
            # Skip critical edges
            if edge in critical_edges or (edge[1], edge[0]) in critical_edges:
                continue
            
            # Check if removing edge would disconnect the graph
            G_copy.remove_edge(*edge)
            
            # If removing the edge disconnects the graph, add it back
            if not nx.is_connected(G_copy):
                G_copy.add_edge(*edge)
                continue
            
            # Check if removing edge significantly increases path lengths for AGVs
            significant_increase = False
            for agv in self.agvs:
                if not agv.delivered and agv.targets:
                    source = (round(agv.pos[0]), round(agv.pos[1]))
                    for target in agv.targets:
                        try:
                            original_path_length = len(nx.shortest_path(self.G, source, target))
                            new_path_length = len(nx.shortest_path(G_copy, source, target))
                            
                            # If path length increases by more than 30%, consider it significant
                            if new_path_length > original_path_length * 1.3:
                                significant_increase = True
                                break
                        except nx.NetworkXNoPath:
                            significant_increase = True
                            break
                    if significant_increase:
                        break
            
            # If removing the edge causes significant issues, add it back
            if significant_increase:
                G_copy.add_edge(*edge)
            else:
                removed_count += 1
                
                # Remove congestion data for eliminated edge
                if edge in self.congestion_data:
                    del self.congestion_data[edge]
                if (edge[1], edge[0]) in self.congestion_data:
                    del self.congestion_data[(edge[1], edge[0])]
                
                # Limit the number of edges removed at once to avoid over-optimization
                if removed_count >= len(self.G.edges()) // 10:  # Remove at most 10% of edges
                    break
        
        # Update the graph
        self.G = G_copy
        self.message = f"Removed {removed_count} redundant arcs"

    def identify_flow_concentrated_cells(self):
        """
        Identifies high-traffic zones (cells) in the network where flow is densely concentrated
        """
        if not self.G.edges() or not self.agvs:
            return
        
        # Initialize flow data based on current AGV paths and history
        self.initialize_flow_data()
        
        # Update flow data based on AGV paths and history
        for agv in self.agvs:
            # Add flow from current path
            if agv.path:
                for i in range(len(agv.path)-1):
                    u, v = agv.path[i], agv.path[i+1]
                    self.flow_data[(u, v)] += 1
                    self.flow_data[(v, u)] += 1
            
            # Add flow from path history
            path_list = list(agv.path_history)
            for i in range(len(path_list)-1):
                u = (round(path_list[i][0]), round(path_list[i][1]))
                v = (round(path_list[i+1][0]), round(path_list[i+1][1]))
                if self.G.has_edge(u, v):
                    self.flow_data[(u, v)] += 0.5  # Historical paths have less weight
                    self.flow_data[(v, u)] += 0.5
        
        # Calculate flow concentration by cell
        cell_flow = defaultdict(float)
        for (u, v), flow_value in self.flow_data.items():
            if u in self.cell_map and v in self.cell_map:
                cell_u = self.cell_map[u]
                cell_v = self.cell_map[v]
                cell_flow[cell_u] += flow_value
                if cell_u != cell_v:
                    cell_flow[cell_v] += flow_value
        
        # Identify highest flow cells (top 30%)
        if not cell_flow:
            return
            
        high_traffic_threshold = sorted(cell_flow.values(), reverse=True)
        if high_traffic_threshold:
            threshold_idx = max(0, int(len(high_traffic_threshold) * 0.3) - 1)
            high_traffic_threshold = high_traffic_threshold[threshold_idx]
        else:
            high_traffic_threshold = 0
            
        self.high_traffic_cells = {cell for cell, flow in cell_flow.items() if flow >= high_traffic_threshold}
        
        # Update all AGVs with the new flow data
        for agv in self.agvs:
            agv.flow_data = self.flow_data
        
        self.message = f"Identified {len(self.high_traffic_cells)} high-traffic cells"

    def dynamic_multicommodity_flow(self):
        """
        Dynamic Multicommodity Flow optimization
        Optimizes the flow of multiple AGVs sharing the network with dynamically changing demands
        """
        if not self.agvs or not self.G.edges():
            return
        
        # Initialize or update flow data
        self.identify_flow_concentrated_cells()
        
        # Current time-step demand for each AGV (source, target)
        demands = []
        for agv in self.agvs:
            if not agv.delivered and agv.targets:
                source = (round(agv.pos[0]), round(agv.pos[1]))
                target = agv.targets[0]
                demands.append((agv.id, source, target))
        
        if not demands:
            return
        
        # Step 1: Calculate current network load
        edge_load = defaultdict(int)
        for agv in self.agvs:
            if agv.path:
                for i in range(len(agv.path)-1):
                    edge = (agv.path[i], agv.path[i+1])
                    edge_load[edge] += 1
                    edge_load[(edge[1], edge[0])] += 1
        
        # Step 2: Calculate edge costs based on current load and congestion
        edge_costs = {}
        for u, v in self.G.edges():
            edge = (u, v)
            rev_edge = (v, u)
            
            # Base cost is 1.0
            base_cost = 1.0
            
            # Load factor increases cost for heavily used edges
            load_factor = 1.0 + (edge_load.get(edge, 0) / 3.0)  # Assume capacity of 3 per edge
            
            # Congestion factor from historical data
            congestion_factor = 1.0 + (self.congestion_data.get(edge, 0) / 10.0)
            
            # Cell-based factor for high traffic areas
            u_cell = self.cell_map.get(u, 0)
            v_cell = self.cell_map.get(v, 0)
            cell_factor = 1.5 if u_cell in self.high_traffic_cells or v_cell in self.high_traffic_cells else 1.0
            
            # Final edge cost
            total_cost = base_cost * load_factor * congestion_factor * cell_factor
            
            edge_costs[edge] = total_cost
            edge_costs[rev_edge] = total_cost
        
        # Step 3: Find new optimal paths for all AGVs based on dynamic costs
        paths_updated = 0
        for agv_id, source, target in demands:
            # Create a copy of the graph with updated edge weights
            G_weighted = nx.Graph()
            for u, v in self.G.edges():
                G_weighted.add_edge(u, v, weight=edge_costs.get((u, v), 1.0))
            
            # Find optimal path with dijkstra
            try:
                new_path = nx.dijkstra_path(G_weighted, source, target, weight='weight')
                
                # Update AGV path
                for agv in self.agvs:
                    if agv.id == agv_id:
                        old_path = agv.path
                        agv.path = new_path[1:]  # Skip the first node (current position)
                        
                        # Check if path actually changed
                        if old_path != agv.path:
                            paths_updated += 1
                        break
            except nx.NetworkXNoPath:
                continue
        
        self.message = f"Dynamic multicommodity flow: updated {paths_updated} paths"

    def multicommodity_flow(self):
        if not self.agvs or not self.G.edges():
            return

        flow = defaultdict(float)
        for agv in self.agvs:
            if agv.path:
                for i in range(len(agv.path)-1):
                    edge = (agv.path[i], agv.path[i+1])
                    flow[edge] += 1
                    flow[(edge[1], edge[0])] += 1

        for agv in self.agvs:
            if agv.targets and not agv.path:
                target = agv.targets[0]
                algorithms = ['astar', 'dynamic_mcf', 'cell_based', 'flow_concentrated']
                best_path = None
                best_cost = float('inf')
                
                for algo in algorithms:
                    path = agv.find_path(agv.pos, target, algorithm=algo)
                    if path:
                        cost = sum(flow.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                
                agv.path = best_path or []

        self.message = "Multicommodity flow optimized"

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def add_agv(self, start_pos, targets):
        # Ensure start_pos is a valid node in the graph
        if not self.G:
            self.message = "Graph not initialized!"
            return
            
        # Round positions to ensure they're valid graph nodes
        start_pos = (round(start_pos[0]), round(start_pos[1]))
        rounded_targets = []
        for target in targets:
            rounded_targets.append((round(target[0]), round(target[1])))
        
        # Check if start position is in graph
        if start_pos not in self.G.nodes():
            # Try to find closest valid node
            closest_node = None
            min_dist = float('inf')
            for node in self.G.nodes():
                dist = abs(node[0] - start_pos[0]) + abs(node[1] - start_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node
                    
            if closest_node and min_dist <= 1.0:
                start_pos = closest_node
                self.message = f"Adjusted start position to nearest valid node {start_pos}"
            else:
                self.message = "Start position not in graph and no nearby valid node found!"
                return

        # Assign color and create AGV
        color = self.agv_colors[(self.next_agv_id-1) % len(self.agv_colors)]
        agv = AGV(self.next_agv_id, start_pos, deque(rounded_targets), color)
        
        # Set references to necessary data structures
        agv.G = self.G
        agv.congestion_data = self.congestion_data
        agv.cell_map = self.cell_map
        agv.flow_data = self.flow_data if hasattr(self, 'flow_data') else None

        # Find initial path
        if rounded_targets:
            try:
                agv.path = agv.find_path(start_pos, rounded_targets[0]) or []
            except Exception as e:
                print(f"Error finding initial path for new AGV: {e}")
                # Try A* as fallback
                try:
                    agv.path = agv.astar_path(start_pos, rounded_targets[0]) or []
                except Exception as e2:
                    print(f"A* fallback also failed: {e2}")
                    agv.path = []

        # Add AGV to simulation
        self.agvs.append(agv)
        self.next_agv_id += 1
        self.message = f"Added AGV {agv.id} with {len(rounded_targets)} targets"
        
        # Store the configuration whenever a new AGV is added
        self.store_current_configuration()

    def update_simulation(self):
        if not self.simulation_running:
            return

        try:
            # Run optimization algorithms periodically
            if self.frame_count % 100 == 0:
                try:
                    self.optimize_guide_path()
                except Exception as e:
                    print(f"Error in optimize_guide_path: {e}")
                
                try:
                    self.eliminate_subtours()
                except Exception as e:
                    print(f"Error in eliminate_subtours: {e}")
                
                # Execute the new algorithms less frequently
                if self.frame_count % 300 == 0:
                    try:
                        self.redundant_arc_elimination()
                    except Exception as e:
                        print(f"Error in redundant_arc_elimination: {e}")
                    
                    try:
                        self.identify_flow_concentrated_cells()
                    except Exception as e:
                        print(f"Error in identify_flow_concentrated_cells: {e}")
                
                try:
                    self.dynamic_multicommodity_flow()
                except Exception as e:
                    print(f"Error in dynamic_multicommodity_flow: {e}")

            self.frame_count += 1

            # Track current positions of all AGVs
            current_positions = defaultdict(list)
            for agv in self.agvs:
                if not agv.delivered:
                    rounded_pos = (round(agv.pos[0]), round(agv.pos[1]))
                    current_positions[rounded_pos].append(agv.id)

            # Check for collisions
            collision_detected = False
            for pos, agv_ids in current_positions.items():
                if len(agv_ids) > 1:
                    collision_detected = True
                    self.collisions += 1
                    for agv_id in agv_ids:
                        for agv in self.agvs:
                            if agv.id == agv_id:
                                agv.collision_count += 1

            # Create occupied positions map
            occupied = {}
            for agv in self.agvs:
                if not agv.delivered:
                    occupied[(round(agv.pos[0]), round(agv.pos[1]))] = agv.id

            # Update each AGV
            for agv in self.agvs:
                if agv.delivered:
                    continue

                # Update position and handle any errors
                try:
                    result = agv.update_position(occupied)
                except Exception as e:
                    print(f"Error updating AGV {agv.id} position: {e}")
                    continue

                # Replan if needed
                if result == "replan" and agv.targets:
                    try:
                        # Use one of the new algorithms for replanning
                        algorithms = ['astar', 'dynamic_mcf', 'cell_based', 'flow_concentrated']
                        algorithm = random.choice(algorithms)
                        agv.path = agv.find_path(agv.pos, agv.targets[0], algorithm=algorithm) or []
                    except Exception as e:
                        print(f"Error replanning path for AGV {agv.id}: {e}")
                        # Fallback to A* if other algorithms fail
                        try:
                            agv.path = agv.astar_path(agv.pos, agv.targets[0]) or []
                        except Exception as e2:
                            print(f"Fallback A* also failed: {e2}")

                # Update congestion and flow data
                if len(agv.path_history) > 1:
                    try:
                        prev_pos = agv.path_history[-2]
                        curr_pos = agv.path_history[-1]
                        prev_pos_rounded = (round(prev_pos[0]), round(prev_pos[1]))
                        curr_pos_rounded = (round(curr_pos[0]), round(curr_pos[1]))
                        
                        # Update congestion data
                        self.congestion_data[(prev_pos_rounded, curr_pos_rounded)] = \
                            self.congestion_data.get((prev_pos_rounded, curr_pos_rounded), 0) + 1
                        self.congestion_data[(curr_pos_rounded, prev_pos_rounded)] = \
                            self.congestion_data.get((curr_pos_rounded, prev_pos_rounded), 0) + 1
                        
                        # Update flow data
                        if hasattr(self, 'flow_data'):
                            self.flow_data[(prev_pos_rounded, curr_pos_rounded)] = \
                                self.flow_data.get((prev_pos_rounded, curr_pos_rounded), 0) + 0.5
                            self.flow_data[(curr_pos_rounded, prev_pos_rounded)] = \
                                self.flow_data.get((curr_pos_rounded, prev_pos_rounded), 0) + 0.5
                    except Exception as e:
                        print(f"Error updating congestion/flow data: {e}")

            # Calculate total wait time
            self.total_wait_time = sum(agv.waiting for agv in self.agvs)

            # Update message for collision
            if collision_detected:
                self.message = f"Collision detected! Total: {self.collisions}"
                
        except Exception as e:
            print(f"Critical error in update_simulation: {e}")
            # Continue simulation despite errors
            self.message = f"Error: {str(e)[:50]}..."

    def draw_grid(self):
        # Draw background gradient
        for y in range(HEIGHT):
            ratio = y / HEIGHT
            r = int(BACKGROUND_START[0] * (1 - ratio) + BACKGROUND_END[0] * ratio)
            g = int(BACKGROUND_START[1] * (1 - ratio) + BACKGROUND_END[1] * ratio)
            b = int(BACKGROUND_START[2] * (1 - ratio) + BACKGROUND_END[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))

        # Draw grid background with border
        grid_rect = pygame.Rect(MARGIN - 5, MARGIN - 5, GRID_SIZE * CELL_SIZE + 10, GRID_SIZE * CELL_SIZE + 10)
        pygame.draw.rect(screen, DARK_GRAY, grid_rect, 2)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    MARGIN + x * CELL_SIZE,
                    MARGIN + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                # Color high traffic cells differently
                cell = self.cell_map.get((x, y), 0)
                if hasattr(self, 'high_traffic_cells') and cell in self.high_traffic_cells:
                    pygame.draw.rect(screen, (255, 240, 240), rect)  # Light red for high traffic
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

        # Draw congestion heatmap
        max_congestion = max(self.congestion_data.values()) if self.congestion_data else 1
        for (u, v), value in self.congestion_data.items():
            if value > 0:
                start = (MARGIN + u[0]*CELL_SIZE + CELL_SIZE//2, 
                         MARGIN + u[1]*CELL_SIZE + CELL_SIZE//2)
                end = (MARGIN + v[0]*CELL_SIZE + CELL_SIZE//2, 
                       MARGIN + v[1]*CELL_SIZE + CELL_SIZE//2)
                intensity = min(255, int(200 * value/max(1, max_congestion)))
                pygame.draw.line(screen, (intensity, 50, 50), start, end, 4)

        # Draw flow data if available
        if hasattr(self, 'flow_data') and self.flow_data:
            max_flow = max(self.flow_data.values()) if self.flow_data else 1
            for (u, v), value in self.flow_data.items():
                if value > 0.5:  # Only draw significant flows
                    start = (MARGIN + u[0]*CELL_SIZE + CELL_SIZE//2, 
                             MARGIN + u[1]*CELL_SIZE + CELL_SIZE//2)
                    end = (MARGIN + v[0]*CELL_SIZE + CELL_SIZE//2, 
                           MARGIN + v[1]*CELL_SIZE + CELL_SIZE//2)
                    intensity = min(255, int(200 * value/max(1, max_flow)))
                    pygame.draw.line(screen, (50, 50, intensity), start, end, 2)

        # Draw graph edges
        for (x1, y1), (x2, y2) in self.G.edges():
            start = (MARGIN + x1*CELL_SIZE + CELL_SIZE//2, 
                     MARGIN + y1*CELL_SIZE + CELL_SIZE//2)
            end = (MARGIN + x2*CELL_SIZE + CELL_SIZE//2, 
                   MARGIN + y2*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.line(screen, DARK_GRAY, start, end, 3)

        # Draw nodes (larger with shadow)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                center = (MARGIN + x*CELL_SIZE + CELL_SIZE//2,
                          MARGIN + y*CELL_SIZE + CELL_SIZE//2)
                # Shadow
                pygame.draw.circle(screen, SHADOW_COLOR, (center[0] + 3, center[1] + 3), 15)
                # Node
                color = BLUE if (x,y) in self.selected_nodes else GRAY
                pygame.draw.circle(screen, color, center, 15)
                pygame.draw.circle(screen, BLACK, center, 15, 2)
                node_num = y*GRID_SIZE + x
                num_text = font.render(str(node_num), True, BLACK)
                screen.blit(num_text, (center[0]-8, center[1]-8))

        # Draw AGV paths
        for agv in self.agvs:
            if len(agv.path_history) > 1:
                points = []
                for pos in agv.path_history:
                    points.append((
                        MARGIN + pos[0]*CELL_SIZE + CELL_SIZE//2,
                        MARGIN + pos[1]*CELL_SIZE + CELL_SIZE//2
                    ))
                if len(points) > 1:
                    pygame.draw.lines(screen, agv.color, False, points, 3)

        # Draw AGVs as bot-like robots
        for agv in self.agvs:
            if not agv.delivered:
                agv_x = int(MARGIN + agv.pos[0]*CELL_SIZE + CELL_SIZE//2)
                agv_y = int(MARGIN + agv.pos[1]*CELL_SIZE + CELL_SIZE//2)
                # Shadow for depth
                pygame.draw.rect(screen, SHADOW_COLOR, (agv_x - 15 + 3, agv_y - 15 + 3, 30, 30))
                # Body (rectangle)
                pygame.draw.rect(screen, agv.color, (agv_x - 15, agv_y - 15, 30, 30))
                pygame.draw.rect(screen, BLACK, (agv_x - 15, agv_y - 15, 30, 30), 2)
                # Wheels
                pygame.draw.circle(screen, BLACK, (agv_x - 10, agv_y + 15), 5)
                pygame.draw.circle(screen, BLACK, (agv_x + 10, agv_y + 15), 5)
                # Antenna
                pygame.draw.line(screen, BLACK, (agv_x, agv_y - 15), (agv_x, agv_y - 25), 2)
                pygame.draw.circle(screen, BLACK, (agv_x, agv_y - 25), 3)
                # ID label
                id_text = font.render(str(agv.id), True, WHITE)
                screen.blit(id_text, (agv_x - 5, agv_y - 5))

        # Draw GUI panels
        # Stats panel (top-right)
        stats_panel = pygame.Surface((250, 180), pygame.SRCALPHA)
        stats_panel.fill((255, 255, 255, 240))  # Semi-transparent white
        pygame.draw.rect(stats_panel, DARK_GRAY, (0, 0, 250, 180), 2, border_radius=10)
        screen.blit(stats_panel, (WIDTH - 260, 10))
        # Shadow for stats panel
        shadow = pygame.Surface((250, 180), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 50))
        screen.blit(shadow, (WIDTH - 260 + 5, 15))

        stats = [
            f"Active AGVs: {len([a for a in self.agvs if not a.delivered])}/{len(self.agvs)}",
            f"Collisions: {self.collisions}",
            f"Wait Time: {self.total_wait_time}",
            f"Congestion: {sum(self.congestion_data.values())}",
            f"High Traffic Cells: {len(getattr(self, 'high_traffic_cells', []))}"
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, BLACK)
            screen.blit(text, (WIDTH - 250, 20 + i*30))

        # Controls panel (below stats panel)
        controls_panel = pygame.Surface((250, 320), pygame.SRCALPHA)
        controls_panel.fill((255, 255, 255, 240))
        pygame.draw.rect(controls_panel, DARK_GRAY, (0, 0, 250, 320), 2, border_radius=10)
        screen.blit(controls_panel, (WIDTH - 260, 200))
        # Shadow for controls panel
        shadow = pygame.Surface((250, 320), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 50))
        screen.blit(shadow, (WIDTH - 260 + 5, 205))

        controls = [
            "Controls:",
            "Click: Select nodes",
            "A: Add AGV to path",
            "P: Add path arcs",
            "O: Optimize guide path",
            "S: Eliminate subtours",
            "F: Run multicommodity flow",
            "R: Reset simulation",
            "C: Clear congestion",
            "D: Dynamic MCF",
            "E: Redundant arc elimination",
            "I: Identify flow-concentrated cells",
            "L: Replay last configuration",
            "Space: Start/pause",
            "Q: Quit"
        ]
        for i, control in enumerate(controls):
            text = font.render(control, True, BLACK)
            screen.blit(text, (WIDTH - 250, 210 + i*22))

        # Status bar (with simulation state)
        status_panel = pygame.Surface((WIDTH, 40))
        status_panel.fill(DARK_GRAY)
        screen.blit(status_panel, (0, HEIGHT - 40))
        state_text = "Running" if self.simulation_running else "Paused"
        status_message = f"{self.message} | {state_text}"
        status_text = large_font.render(status_message, True, WHITE)
        screen.blit(status_text, (10, HEIGHT - 35))

        # Title (with background panel)
        title_panel = pygame.Surface((300, 40), pygame.SRCALPHA)
        title_panel.fill((255, 255, 255, 240))
        pygame.draw.rect(title_panel, DARK_GRAY, (0, 0, 300, 40), 2, border_radius=5)
        screen.blit(title_panel, (10, 10))
        title_text = title_font.render("AGV Path Optimization", True, BLACK)
        screen.blit(title_text, (15, 12))

    def handle_click(self, pos):
        x = (pos[0] - MARGIN) // CELL_SIZE
        y = (pos[1] - MARGIN) // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            node = (x, y)
            if node not in self.selected_nodes:
                self.selected_nodes.append(node)
                self.message = f"Selected node {y*GRID_SIZE + x}"
            else:
                self.selected_nodes.remove(node)

    def add_path_arcs(self):
        if len(self.selected_nodes) >= 2:
            for i in range(len(self.selected_nodes)-1):
                u, v = self.selected_nodes[i], self.selected_nodes[i+1]
                if not self.G.has_edge(u, v):
                    self.G.add_edge(u, v)
                    self.congestion_data[(u,v)] = 0
                    self.congestion_data[(v,u)] = 0
                    if hasattr(self, 'flow_data'):
                        self.flow_data[(u,v)] = 0
                        self.flow_data[(v,u)] = 0
            self.message = f"Added path with {len(self.selected_nodes)-1} segments"
            self.selected_nodes = []

    def clear_congestion(self):
        self.initialize_congestion_data()
        if hasattr(self, 'flow_data'):
            self.initialize_flow_data()
        self.message = "Congestion data cleared"

    def reset(self):
        last_config = self.last_configuration  # Preserve last configuration
        self.G = self.create_grid_graph()
        self.agvs = []
        self.selected_nodes = []
        self.collisions = 0
        self.total_wait_time = 0
        self.next_agv_id = 1
        self.simulation_running = False
        self.message = "Simulation reset"
        self.initialize_congestion_data()
        self.initialize_flow_data()
        self.cell_map = self.create_cell_map()
        self.high_traffic_cells = set()
        self.agv_colors = [
            RED, GREEN, BLUE, PURPLE, ORANGE, 
            CYAN, YELLOW, DARK_GREEN
        ]
        self.frame_count = 0
        self.last_configuration = last_config  # Restore last configuration

def main():
    simulator = AGVSimulator()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                simulator.handle_click(event.pos)
            elif event.type == KEYDOWN:
                if event.key == pygame.K_a:
                    if len(simulator.selected_nodes) >= 2:
                        simulator.add_agv(
                            simulator.selected_nodes[0], 
                            simulator.selected_nodes[1:]
                        )
                        simulator.selected_nodes = []
                elif event.key == pygame.K_p:
                    simulator.add_path_arcs()
                    simulator.selected_nodes = []
                elif event.key == pygame.K_o:
                    simulator.optimize_guide_path()
                elif event.key == pygame.K_s:
                    simulator.eliminate_subtours()
                elif event.key == pygame.K_f:
                    simulator.multicommodity_flow()
                elif event.key == pygame.K_d:
                    simulator.dynamic_multicommodity_flow()
                elif event.key == pygame.K_e:
                    simulator.redundant_arc_elimination()
                elif event.key == pygame.K_i:
                    simulator.identify_flow_concentrated_cells()
                elif event.key == pygame.K_l:  # New replay functionality
                    simulator.replay_last_configuration()
                elif event.key == pygame.K_SPACE:
                    simulator.simulation_running = not simulator.simulation_running
                elif event.key == pygame.K_r:
                    simulator.reset()
                elif event.key == pygame.K_c:
                    simulator.clear_congestion()
                elif event.key == pygame.K_q:
                    running = False

        simulator.update_simulation()
        simulator.draw_grid()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()