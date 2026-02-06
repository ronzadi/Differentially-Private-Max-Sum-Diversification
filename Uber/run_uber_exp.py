import pandas as pd
import numpy as np
import folium
from prep import UberOptimizer
from classes import MSDFacilityLocation, GroundSet
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline


def run_uber_experiment(objective, ground_set, passenger_coords, grid_coords, params, rep):
    """
    Executes all algorithm variants and generates results/visualizations.
    """
    results = []
    k, eps, p, lam, g = params['k'], params['eps'], params['private'], params['lambda'], params['gamma']

    # --- Algorithm Suite Setup ---
    # format: (label, function, args)
    algorithms = [
        ('nonpriv', greedy, [objective, ground_set, k]),
        ('DPGreedy', DP_greedy, [objective, ground_set, k, eps, p]),
        ('DPSampleOblGreedy', DP_sample_greedy, [objective, ground_set, k, eps, p, True, g]),
        ('DPSampleGreedy', DP_sample_greedy, [objective, ground_set, k, eps, p, False, g]),
        ('Random', random_baseline, [objective, ground_set, k])
    ]

    # --- Execute and Collect Results ---
    final_selected = {}
    for i in range(rep):
        for name, func, args in algorithms:
            print(f"--- Running {name} ---")
            # Handle different return signatures (Random doesn't return queries)
            res = func(*args)
            selected, value = res[0], res[1]
            queries = res[2] if len(res) > 2 else 0

            print(f"{name} Value: {value:.4f}")

            results.append({
                'alg': name, 'k': k, 'lambda_param': lam, 'eps': eps,
                'private': p, 'selected': selected, 'value': value, 'queries': queries
            })
            final_selected[name] = selected

    # --- Save Results ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"Uber_results_{params['n_locs']}_{params['spurious']}.csv", index=False)

    # --- Visualization ---
    m = folium.Map(location=[40.78, -73.97], zoom_start=12, tiles='CartoDB positron')

    # Groups
    groups = {
        'All Passengers': folium.FeatureGroup(name='All Passengers').add_to(m),
        'Candidates': folium.FeatureGroup(name='Candidates (Yellow)').add_to(m),
        'Non-Private Greedy': folium.FeatureGroup(name='Non-Private (Red)').add_to(m),
        'DP Greedy': folium.FeatureGroup(name='DP Greedy (Brown)').add_to(m)
    }

    # Plot Passengers
    for lat, lon in passenger_coords:
        folium.CircleMarker([lat, lon], radius=0.5, color='blue', fill=True).add_to(groups['All Passengers'])

    # Plot All Grid Candidates
    for lat, lon in grid_coords:
        folium.CircleMarker([lat, lon], radius=2, color='yellow', fill=True, opacity=0.4).add_to(groups['Candidates'])

    # Plot Selected Hubs (Non-Private)
    for idx in final_selected['nonpriv']:
        folium.CircleMarker(grid_coords[idx], radius=5, color='red', fill=True).add_to(groups['Non-Private Greedy'])

    # Plot Selected Hubs (DP Greedy)
    for idx in final_selected['DPGreedy']:
        folium.CircleMarker(grid_coords[idx], radius=5, color='brown', fill=True).add_to(groups['DP Greedy'])

    folium.LayerControl().add_to(m)
    m.save("Uber_results.html")

    return final_selected['DPGreedy']


# --- Execution ---
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'k': 5,
        'eps': 0.1,
        'lambda': 0.1,
        'private': True,
        'gamma': 0.1,
        'n_locs': 1000,
        'spurious': 500
    }

    DATA_PATH = r"C:\Users\Ronza\Dev\DP-MSD\Uber\tmp.csv"

    HULL = [
        (40.7005038, -74.0144209), (40.7112088, -73.9776851), (40.7282434, -73.9720702),
        (40.7418214, -73.9733576), (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482), (40.8769142, -73.9269985),
        (40.8512745, -73.9448513), (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    # 1. Pre-process
    opt = UberOptimizer(HULL, n_data=np.inf)
    passengers = opt.process_raw_data(DATA_PATH, "sampled_passengers.csv")
    grid = opt.create_grid(n_locs=CONFIG['n_locs'], spurious=CONFIG['spurious'])

    # 2. Setup Problem Objects
    obj = MSDFacilityLocation(
        passenger_coords=passengers,
        grid_coords=grid,
        lambda_param=CONFIG['lambda'],
        k=CONFIG['k'],
        distortion=0,
        sensitivity=1 / len(passengers)
    )
    g_set = GroundSet(elements=list(range(len(grid))))

    # 3. Run Experiment
    hubs = run_uber_experiment(obj, g_set, passengers, grid, CONFIG, 10)