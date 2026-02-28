import math
import os
import platform
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Custom module imports
from prep import UberOptimizer
from classes import MSDUberObjective, GroundSet
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline


def get_best_eps_0(eps_target: float, delta_target: float, k: int) -> float:
    """
    Calculates privacy budget per step (epsilon_0) by taking the maximum
    permissible value across Basic, Advanced, and Gupta composition bounds.
    """
    # 1. Basic Composition
    eps_basic = eps_target / k

    # 2. Advanced Composition (Dwork et al.)
    # Solving for eps_0 in: eps = sqrt(2k ln(1/delta))eps_0 + k*eps_0(e^eps_0 - 1)
    # This is a common approximation used in DP literature
    term1 = (2 * math.log(1.0 / delta_target)) / k
    eps_adv = math.sqrt(term1 + (eps_target / k)) - math.sqrt(term1)

    # 3. Gupta Bound (Specifically for decomposable objectives)
    eps_gupta = math.log(1 + eps_target / (4 + math.log(1.0 / delta_target)))

    return max(eps_basic, eps_adv, eps_gupta)


def run_uber_experiment(
        objective: MSDUberObjective,
        ground_set: GroundSet,
        passenger_coords: List[tuple],
        grid_coords: List[tuple],
        params: Dict[str, Any],
        iterations: int
) -> List[int]:
    all_runs_data = []
    rk, eps = params['k'], params['eps']
    is_private, lam, gamma = params['private'], params['lambda'], params['gamma']

    # Define delta based on dataset size: 1/m^1.5
    delta_target = 1 / (len(passenger_coords) ** 1.5)
    eps_0 = get_best_eps_0(eps_target=eps, delta_target=delta_target, k=rk)

    # Algorithm configurations: (Label, Function, Arguments)
    algorithms = [
        ('Non-Private', DP_greedy, [objective, ground_set, rk, eps_0, False]),
        ('DP-Greedy', DP_greedy, [objective, ground_set, rk, eps_0, is_private]),
        ('DP-Sample-Oblivious', DP_sample_greedy, [objective, ground_set, rk, eps_0, is_private, True, gamma]),
        ('DP-Sample-Greedy', DP_sample_greedy, [objective, ground_set, rk, eps_0, is_private, False, gamma]),
        ('Random', random_baseline, [objective, ground_set, rk])
    ]

    dp_greedy_selection = []

    for run_idx in range(iterations):
        for name, alg_func, args in algorithms:
            print(f"  [Executing] {name}...")

            start_clock = time.time()
            res = alg_func(*args)
            execution_time = time.time() - start_clock

            # Standardizing result unpacking
            selected_indices = res[0]
            total_val, relevance, diversity = res[1], res[2], res[3]
            query_count = res[4] if len(res) > 4 else 0

            all_runs_data.append({
                'algorithm': name,
                'k': rk,
                'lambda': lam,
                'epsilon_total': eps,
                'epsilon_step': round(eps_0, 6),
                'iteration': run_idx,
                'obj_value': total_val,
                'relevance': relevance,
                'diversity': diversity,
                'queries': query_count,
                'runtime_sec': round(execution_time, 4)
            })

            if name == 'DP-Greedy':
                dp_greedy_selection = selected_indices

    results_df = pd.DataFrame(all_runs_data)
    csv_path = f"results/Uber_Master_Results_{params['n_locs']}_{params['spurious']}.csv"

    os.makedirs('results', exist_ok=True)
    header_needed = not os.path.exists(csv_path)
    results_df.to_csv(csv_path, mode='a', index=False, header=header_needed)

    return dp_greedy_selection


if __name__ == "__main__":
    base_dir = '../' if platform.system() == 'Windows' else ''
    DATA_PATH = os.path.join(base_dir, 'datasets', 'uber', 'passengers.csv')

    # Manhattan Convex Hull
    MANHATTAN_HULL = [
        (40.7005, -74.0144), (40.7112, -73.9777), (40.7282, -73.9721),
        (40.7418, -73.9734), (40.7755, -73.9430), (40.7975, -73.9297),
        (40.8351, -73.9354), (40.8713, -71.9109), (40.8769, -73.9270),
        (40.8513, -73.9449), (40.7608, -74.0041), (40.7474, -74.0115),
        (40.7126, -74.0182)
    ]

    # Parameter Sweep Definition
    # --- Professional Parameter Grid (Faithful to Original) ---

    k_sweep = [
        {'k': k, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.2, 'n_locs': 1000, 'spurious': 800}
        for k in [4, 6, 8, 10, 12, 14, 16, 18, 20, 30]
    ]

    eps_sweep = [
        {'k': 6, 'eps': e, 'lambda': 0.1, 'private': True, 'gamma': 0.2, 'n_locs': 1000, 'spurious': 800}
        for e in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    ]

    # 3. Lambda Sweep: k=6, eps=0.2, varying lambda from 0 to 0.8
    lambda_sweep = [
        {'k': 6, 'eps': 0.2, 'lambda': l, 'private': True, 'gamma': 0.2, 'n_locs': 1000, 'spurious': 800}
        for l in [0, 0.2, 0.4, 0.6, 0.8]
    ]

    # Combine all logic into the final grid
    param_grid = k_sweep + eps_sweep + lambda_sweep
    # Data Initialization
    optimizer = UberOptimizer(MANHATTAN_HULL, n_data=10000)
    passengers = optimizer.read_from_file(DATA_PATH)
    print(f"Dataset loaded: {len(passengers)} passenger records.")

    for config in param_grid:
        print(f"\n{'=' * 40}\nCONFIG: k={config['k']}, eps={config['eps']}, lambda={config['lambda']}\n{'=' * 40}")

        # 1. Spatial Grid Construction
        grid = optimizer.create_grid(n_locs=config['n_locs'], spurious=config['spurious'])

        # 2. Problem Initialization
        # Sensitivity is 1/n because we normalize the relevance function
        objective = MSDUberObjective(
            passenger_coords=passengers,
            grid_coords=grid,
            lambda_param=config['lambda'],
            k=config['k'],
            distortion=0,
            sensitivity=1.0 / len(passengers)
        )
        ground_set = GroundSet(elements=list(range(len(grid))))

        # 3. Execution
        run_uber_experiment(objective, ground_set, passengers, grid, config, iterations=10)