import networkx as nx
import pulp
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Sample Data Definition
nodes = [0, 1, 2, 3]  # 2x2 grid: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
arcs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)]  # Bidirectional arcs
self_loops = [(i, i) for i in nodes]  # Self-loops for waiting
arcs += self_loops
demands = {1: {'o_k': 0, 'd_k': 3, 'f_t': [1.0] + [0.0]*9, 'e_t': [0.0]*9 + [1.0]}}  # Flow from node 0 to 3
capacities = {(i, j): 1.0 for i, j in arcs}  # Arc capacities
fixed_costs = {(i, j): 1.0 if i != j else 0.0 for i, j in arcs}  # Fixed costs
travel_costs = {(i, j): 1.0 for i, j in arcs}  # Travel costs
time_horizon = 10
w1, w2 = 1.0, 1.0  # Weights for objective function
max_iterations = 5
cells = [((0, 1, 2, 3), tuple([(0, 1), (1, 3), (3, 2), (2, 0)]))]  # Cell with edges as tuple

# Create graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(arcs)

# Algorithm 2: Redundant Elimination of Arcs
def redundant_elimination_of_arcs(G, flow_solution, demands, time_horizon):
    print("  Redundant Arc Elimination")
    e = set()  # Arcs with zero flow
    R = set()  # Nodes to check for connectivity
    
    for i, j in G.edges:
        total_flow = sum(flow_solution.get((k, i, j, t), 0.0) for k in demands for t in range(time_horizon))
        if total_flow == 0:
            e.add((i, j))
            print(f"    Arc ({i}, {j}) has zero flow")
    
    for i, j in e:
        predecessors = [l for l in G.predecessors(i) if (l, i) in G.edges and (l, i) not in e]
        if not predecessors:
            R.add(i)
            print(f"    Node {i} added to R (no non-zero flow predecessors)")
        
        successors = [l for l in G.successors(j) if (j, l) in G.edges and (j, l) not in e]
        if len(successors) >= 2:
            R.add(j)
            print(f"    Node {j} added to R (multiple non-zero flow successors)")
    
    G_new = G.copy()
    for i, j in e:
        if i not in R and j not in R:
            G_new.remove_edge(i, j)
            print(f"    Removed arc ({i}, {j})")
    
    if not nx.is_strongly_connected(G_new):
        print("    Warning: Graph is not strongly connected, reverting changes")
        G_new = G.copy()
    
    print(f"    Updated guide path: {list(G_new.edges)}")
    return G_new

# Algorithm 3: Selection of a Key Cell
def select_key_cell(G, cells=cells, flow_solution=None, demands=None, time_horizon=10):
    print("  Key Cell Selection")
    if not cells:
        print("    No cells available")
        return None
    
    if flow_solution and demands:
        cell_flows = {}
        for cell_id, cell_edges in cells:
            total_flow = sum(flow_solution.get((k, i, j, t), 0.0)
                            for k in demands for i, j in cell_edges for t in range(time_horizon))
            cell_flows[cell_id] = total_flow
            print(f"    Cell {cell_id}: Total flow = {total_flow:.2f}")
        
        key_cell = max(cell_flows, key=cell_flows.get, default=random.choice(cells)[0])
        print(f"    Selected key cell (max flow): {key_cell}")
    else:
        key_cell = random.choice(cells)[0]
        print(f"    Selected key cell (random): {key_cell}")
    
    return key_cell

# Algorithm 4: Subtour Elimination
def subtour_elimination(G):
    print("  Subtour Elimination")
    G_new = G.copy()
    
    sccs = list(nx.strongly_connected_components(G_new))
    if len(sccs) <= 1:
        print("    No subtours found")
        return G_new
    
    print(f"    Found {len(sccs)} strongly connected components: {sccs}")
    
    largest_scc = max(sccs, key=len)
    print(f"    Keeping largest component: {largest_scc}")
    
    edges_to_remove = [(u, v) for u, v in G_new.edges if u not in largest_scc or v not in largest_scc]
    for u, v in edges_to_remove:
        G_new.remove_edge(u, v)
        print(f"    Removed edge ({u}, {v})")
    
    print(f"    Updated guide path: {list(G_new.edges)}")
    return G_new

# Helper function to solve dynamic multicommodity flow
def solve_dynamic_multicommodity_flow(G, demands, capacities, time_horizon):
    print("  Solving Dynamic Multicommodity Flow")
    model = pulp.LpProblem("Dynamic_Multicommodity_Flow", pulp.LpMinimize)
    x = {(k, i, j, t): pulp.LpVariable(f"x_{k}_{i}_{j}_{t}", lowBound=0) for k in demands
         for i, j in G.edges for t in range(time_horizon)}
    b = {(k, t): pulp.LpVariable(f"b_{k}_{t}", lowBound=0) for k in demands for t in range(time_horizon)}
    
    model += pulp.lpSum(travel_costs[(i, j)] * x[(k, i, j, t)] for k in demands
                        for i, j in G.edges for t in range(time_horizon))
    
    for k in demands:
        for i in G.nodes:
            for t in range(time_horizon):
                incoming = pulp.lpSum(x[(k, j, i, t)] for j in G.predecessors(i) if (j, i) in G.edges)
                outgoing = pulp.lpSum(x[(k, i, j, t+1)] for j in G.successors(i) if (i, j) in G.edges and t+1 < time_horizon)
                if i == demands[k]['d_k']:
                    model += incoming - outgoing == -b[(k, t)] + (b[(k, t+1)] if t+1 < time_horizon else 0) + demands[k]['e_t'][t]
                elif i == demands[k]['o_k']:
                    model += incoming - outgoing == -demands[k]['f_t'][t]
                else:
                    model += incoming - outgoing == 0
    
    for i, j in G.edges:
        for t in range(time_horizon):
            model += pulp.lpSum(x[(k, i, j, t)] for k in demands) <= capacities[(i, j)]
    
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    if model.status != pulp.LpStatusOptimal:
        raise ValueError("    No feasible solution found")
    
    flow_solution = {(k, i, j, t): pulp.value(x[(k, i, j, t)]) for k, i, j, t in x}
    flow_cost = pulp.value(model.objective)
    print(f"    Flow cost: {flow_cost:.2f}")
    return flow_solution, flow_cost

# Helper function to generate initial guide path
def generate_initial_guide_path(G):
    print("  Generating Initial Guide Path")
    path = nx.DiGraph()
    path.add_edges_from([(0, 1), (1, 3), (3, 3), (0, 0)])  # Minimal connected path
    print(f"    Initial guide path: {list(path.edges)}")
    return path

# Helper function to generate k-neighborhood
def generate_k_neighborhood(key_cell, k, G):
    print(f"  Generating k-Neighborhood (k={k})")
    if key_cell == cells[0][0]:
        neighborhood = [((0, 1, 2, 3), tuple([(0, 1), (1, 3), (3, 2), (2, 0)]))]
        print(f"    k-neighborhood: {neighborhood}")
        return neighborhood
    print("    No neighbors found")
    return []

# Helper function to update guide path
def update_guide_path(current_path, k_cell):
    print("  Updating Guide Path")
    new_path = current_path.copy()
    _, edges = k_cell
    new_path.add_edges_from(edges)
    print(f"    Updated guide path: {list(new_path.edges)}")
    return new_path

# Algorithm 1: Cell-Based Local Search Heuristic
def cell_based_local_search(G, demands, capacities, fixed_costs, travel_costs, time_horizon, max_iterations):
    print("Starting Cell-Based Local Search Heuristic")
    G_current = generate_initial_guide_path(G)
    G_best = G_current.copy()
    f_G_best = float('inf')
    k = 1
    C_neighbor = set()
    iteration = 0
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}")
        is_update = False
        
        try:
            flow_solution, flow_cost = solve_dynamic_multicommodity_flow(G_current, demands, capacities, time_horizon)
        except ValueError as e:
            print(f"Error: {e}")
            break
        
        G_current = redundant_elimination_of_arcs(G_current, flow_solution, demands, time_horizon)
        
        fixed_cost = sum(fixed_costs[(i, j)] for i, j in G_current.edges)
        f_G = w1 * fixed_cost + w2 * flow_cost
        print(f"  Objective value: {f_G:.2f}")
        
        if f_G < f_G_best:
            G_best = G_current.copy()
            f_G_best = f_G
            is_update = True
            print("  New best solution found")
        else:
            G_current = G_best.copy()
        
        if is_update:
            k = 1
            C_neighbor = set()
            c_key = select_key_cell(G_current, cells, flow_solution, demands, time_horizon)
            if c_key is None:
                print("  No key cell selected, stopping")
                break
        
        if not C_neighbor:
            try:
                C_neighbor = set(generate_k_neighborhood(c_key, k, G))
            except TypeError as e:
                print(f"  Error generating k-neighborhood: {e}")
                break
        
        if C_neighbor:
            C_k = C_neighbor.pop()
            G_current = update_guide_path(G_current, C_k)
        else:
            k += 1
            if k > 4:
                print("  Max neighborhood size reached")
                break
        
        G_current = subtour_elimination(G_current)
        
        iteration += 1
    
    print(f"\nFinal best guide path: {list(G_best.edges)}")
    print(f"Final objective value: {f_G_best:.2f}")
    
    # Enhanced Visualization
    pos = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}  # Adjusted for 2x2 grid
    plt.figure(figsize=(8, 8))
    
    # Draw original graph (background)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, label='Nodes')
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=True, label='All Arcs')
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Draw final guide path (foreground)
    nx.draw_networkx_edges(G_best, pos, edge_color='red', width=3, arrows=True, label='Guide Path')
    
    # Add edge labels (optional, for fixed costs)
    edge_labels = {(i, j): f'{fixed_costs[(i, j)]:.1f}' for i, j in G_best.edges}
    nx.draw_networkx_edge_labels(G_best, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("AGV Guide Path Design\n(Gray: All Arcs, Red: Final Guide Path)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('guide_path_output.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'guide_path_output.png'")
    plt.show()
    
    return G_best, f_G_best

# Main Execution
if __name__ == "__main__":
    print("Running AGV Guide Path Design Optimization")
    start_time = time.time()
    try:
        best_path, best_obj = cell_based_local_search(G, demands, capacities, fixed_costs, travel_costs, time_horizon, max_iterations)
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")