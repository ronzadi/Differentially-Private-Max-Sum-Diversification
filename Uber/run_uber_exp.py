import math
import os
import platform
import time

import pandas as pd
import numpy as np
# import folium
from prep import UberOptimizer
from classes import MSDUberObjective, GroundSet
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline


def get_best_eps_0(eps_target, delta_target, k):
    """
    Calculates Basic, Advanced, and Gupta bounds and selects the maximum epsilon_0.
    """
    # 1. Basic Composition
    eps_basic = eps_target / k

    # 2. Advanced Composition
    term1 = (2 * math.log(1.0 / delta_target)) / k
    eps_adv = math.sqrt(term1 + (eps_target / k)) - math.sqrt(term1)

    # 3. Gupta Bound (For decomposable objectives)
    eps_gupta = math.log(1 + eps_target/(3 + math.log(1.0 / delta_target)))

    return max(eps_basic, eps_adv, eps_gupta)

def run_uber_experiment(objective, ground_set, passenger_coords, grid_coords, params, rep):
    """
    Executes all algorithm variants and generates results/visualizations.
    """
    results = []
    k, eps, p, lam, g = params['k'], params['eps'], params['private'], params['lambda'], params['gamma']

    delta_target = 1/(len(passenger_coords))**1.5
    eps_0 = get_best_eps_0(eps_target=eps, delta_target=delta_target, k=k)

    # --- Algorithm Suite Setup ---
    # format: (label, function, args)
    algorithms = [
        # ('nonpriv', greedy, [objective, ground_set, k]),
        ('nonpriv', DP_greedy, [objective, ground_set, k, eps_0, False]),
        ('DPGreedy', DP_greedy, [objective, ground_set, k, eps_0, p]),
        ('DPSampleOblGreedy', DP_sample_greedy, [objective, ground_set, k, eps_0, p, True, g]),
        ('DPSampleGreedy', DP_sample_greedy, [objective, ground_set, k, eps_0, p, False, g]),
        ('Random', random_baseline, [objective, ground_set, k])
    ]

    # --- Execute and Collect Results ---
    final_selected = {}
    for i in range(rep):
        for name, func, args in algorithms:
            print(name)
            # --- Timing Start ---
            start_time = time.time()
            res = func(*args)
            end_time = time.time()
            duration = end_time - start_time
            # --- Timing End ---
            selected, value, rel, div = res[0], res[1], res[2], res[3]
            queries = res[4] if len(res) > 4 else 0

            results.append({
                'alg': name,
                'k': k,
                'lambda_param': lam,
                'eps': eps,
                'rep': i,
                'value': value,
                'relevance': rel,
                'diversity': div,
                'queries': queries,
                'time_sec': round(duration, 4)  # Added time tracking
            })
            final_selected[name] = selected
    # --- Save Results ---
    df_results = pd.DataFrame(results)
    output_file = f"results/Uber_Master_Results_{params['n_locs']}_{params['spurious']}.csv"

    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(output_file)
    df_results.to_csv(output_file, mode='a', index=False, header=not file_exists)

    # Also save unique map for each config so they don't overwrite
    # df_results = pd.DataFrame(results)
    # df_results.to_csv(f"Uber_results_{params['n_locs']}_{params['spurious']}.csv", index=False)

    # --- Visualization ---
    # m = folium.Map(location=[40.78, -73.97], zoom_start=12, tiles='CartoDB positron')
    #
    # # Groups
    # groups = {
    #     'All Passengers': folium.FeatureGroup(name='All Passengers').add_to(m),
    #     'Candidates': folium.FeatureGroup(name='Candidates (Yellow)').add_to(m),
    #     'Non-Private Greedy': folium.FeatureGroup(name='Non-Private (Red)').add_to(m),
    #     'DP Greedy': folium.FeatureGroup(name='DP Greedy (Brown)').add_to(m),
    #     'Random': folium.FeatureGroup(name='Random').add_to(m)
    # }
    #
    # # Plot Passengers
    # for lat, lon in passenger_coords:
    #     folium.CircleMarker([lat, lon], radius=0.5, color='blue', fill=True).add_to(groups['All Passengers'])
    #
    # # Plot All Grid Candidates
    # for lat, lon in grid_coords:
    #     folium.CircleMarker([lat, lon], radius=2, color='pink', fill=True).add_to(groups['Candidates'])
    #
    # # Plot Selected Hubs (Non-Private)
    # for idx in final_selected['nonpriv']:
    #     folium.CircleMarker(grid_coords[idx], radius=5, color='red', fill=True).add_to(groups['Non-Private Greedy'])
    #
    # # Plot Selected Hubs (DP Greedy)
    # for idx in final_selected['DPGreedy']:
    #     folium.CircleMarker(grid_coords[idx], radius=5, color='brown', fill=True).add_to(groups['DP Greedy'])
    #
    # for idx in final_selected['Random']:
    #     folium.CircleMarker(grid_coords[idx], radius=5, color='green', fill=True).add_to(groups['Random'])
    #
    # folium.LayerControl().add_to(m)
    # map_filename = f"Map_k{k}_eps{eps}_lam{lam}.html"
    # m.save(map_filename)
    # # m.save("Uber_results.html")

    return final_selected['DPGreedy']


# --- Execution ---
if __name__ == "__main__":

    prefix = '../' if platform.system() == 'Windows' else ''
    HULL = [ # Manhattan Convex Hull
        (40.7005038, -74.0144209), (40.7112088, -73.9776851), (40.7282434, -73.9720702),
        (40.7418214, -73.9733576), (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482), (40.8769142, -73.9269985),
        (40.8512745, -73.9448513), (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    # Define a list of parameter combinations to test
    param_grid = [
        {'k': 4,  'eps': 0.2, 'lambda': 0.1, 'private': False, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6,  'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 8, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 10, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 12, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 14, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 16, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 18, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 20, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        # {'k': 30, 'eps': 0.1, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        # ###
        {'k': 6, 'eps': 0.02, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.04, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.06, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.08, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.1, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.4, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.6, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.8, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 1, 'lambda': 0.1, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        ####
        {'k': 6, 'eps': 0.2, 'lambda': 0, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.2, 'lambda': 0.2, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.2, 'lambda': 0.4, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.2, 'lambda': 0.6, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
        {'k': 6, 'eps': 0.2, 'lambda': 0.8, 'private': True, 'gamma': 0.1, 'n_locs': 1000, 'spurious': 800},
    ]

    # Initialize pre-processor once
    opt = UberOptimizer(HULL, n_data=10000)
    # passengers = opt.process_raw_data(DATA_PATH, "sampled_passengers.csv")
    passengers = opt.read_from_file(prefix + r'datasets/uber/sampled_passengers89good.csv')
    print(len(passengers))

    for config in param_grid:
        print(f"\n{'=' * 30}\nSTARTING EXPERIMENT: {config}\n{'=' * 30}")

        # 1. Prepare Grid for this specific config
        grid = opt.create_grid(n_locs=config['n_locs'], spurious=config['spurious'])

        # 2. Setup Problem Objects
        obj = MSDUberObjective(
            passenger_coords=passengers,
            grid_coords=grid,
            lambda_param=config['lambda'],
            k=config['k'],
            distortion=0,
            sensitivity=1 / len(passengers)
        )
        g_set = GroundSet(elements=list(range(len(grid))))

        # 3. Run Experiment (updated internal filename logic recommended below)
        hubs = run_uber_experiment(obj, g_set, passengers, grid, config, 10)