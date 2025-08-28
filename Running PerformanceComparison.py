import matplotlib.pyplot as plt
import numpy as np

# Refined synthetic data based on the paper's description
demand_quantities = [2, 25, 50, 75, 78, 100]  # More granular demand range
methods = ['Cell-Based Heuristics', 'CPLEX']

# Computation time (in seconds) for each method across demand quantities
# Cell-Based Heuristics: Starts at 73s, scales better for larger demands
# CPLEX: Faster for small demands, sharp increase for larger demands, fails for demands > 78
comp_times = {
    'Cell-Based Heuristics': [73, 80, 90, 110, 115, 130],
    'CPLEX': [50, 60, 150, 300, None, None]  # None for demands > 78 (CPLEX fails)
}

# Gap in objective values (in %) for the proposed method compared to CPLEX
# CPLEX: 0% gap (optimal) where it solves
# Cell-Based Heuristics: Gap increases to 7% at demand = 78, None for demand > 78
gaps = {
    'Cell-Based Heuristics': [4, 4.5, 5, 6, 7, None],  # None for demand 100 (no CPLEX comparison)
    'CPLEX': [0, 0, 0, 0, 0, None]  # CPLEX is optimal where it finds a solution
}

# Plotting Computation Time with a logarithmic scale for better visualization
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(demand_quantities, comp_times[method], marker='o', label=method)
plt.title('Computation Time vs. Demand Quantity')
plt.xlabel('Demand Quantity')
plt.ylabel('Computation Time (seconds)')
plt.yscale('log')  # Log scale to better visualize the sharp increase in CPLEX time
plt.grid(True, which="both", ls="--")
plt.legend()
# Annotate CPLEX failure point
plt.axvline(x=78, color='red', linestyle='--', label='CPLEX Fails (>78)')
plt.text(78, 50, 'CPLEX Fails', color='red', rotation=90, verticalalignment='bottom')
plt.legend()
plt.savefig('computation_time_cplex_vs_heuristic_refined.png')
plt.close()

# Plotting Gap in Objective Values
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(demand_quantities, gaps[method], marker='o', label=method)
plt.title('Gap in Objective Values vs. Demand Quantity')
plt.xlabel('Demand Quantity')
plt.ylabel('Gap (%)')
plt.grid(True)
# Annotate CPLEX failure point
plt.axvline(x=78, color='red', linestyle='--', label='CPLEX Fails (>78)')
plt.text(78, 1, 'CPLEX Fails', color='red', rotation=90, verticalalignment='bottom')
plt.legend()
plt.savefig('gap_cplex_vs_heuristic_refined.png')
plt.close()