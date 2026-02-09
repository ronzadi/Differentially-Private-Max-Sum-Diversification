import numpy as np
import pandas as pd
import random
from prep import UberOptimizer
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline
import numpy as np
import pandas as pd
from classes import MSDUberObjective, GroundSet
import numpy as np
import pandas as pd
import folium

def run_uber_experiment(k, lambda_param, eps, private, gamma, n_locs, spurious):


    ##################################################  Run Non-Private Greedy ######################################
    nonpriv_selected, nonpriv_value, nonpriv_queries = greedy(objective, ground_set, k)
    print(f"Non-private Greedy Objective Value: {nonpriv_value:.4f}")

    rec = {
        'alg': 'nonpriv',
        'k': k,
        'lambda_param': lambda_param,
        'eps': eps,
        'private': private,
        'selected': nonpriv_selected,
        'value': nonpriv_value,
        'queries': nonpriv_queries
    }

    results.loc[len(results)] = rec


    ##################################################  Run DP Greedy ###############################################
    dp_greedy_selected, dp_greedy_value, dp_greedy_queries = DP_greedy(objective, ground_set, k, eps, private)
    print(f"DP Greedy Objective Value: {dp_greedy_value:.4f}")

    rec = {
        'alg': 'DPGreedy',
        'k': k,
        'lambda_param': lambda_param,
        'eps': eps,
        'private': private,
        'selected': dp_greedy_selected,
        'value': dp_greedy_value,
        'queries': dp_greedy_queries
    }

    results.loc[len(results)] = rec

    ##################################################  Run DP Oblivious Sample Greedy ##############################
    dp_sample_greedy_ob_selected, dp_sample_ob_greedy_value, dp_sample_ob_greedy_queries = DP_sample_greedy(objective, ground_set, k, eps, private, True, gamma)
    print(f"DP Oblivious Sample Greedy Objective Value: {dp_sample_ob_greedy_value:.4f}")

    rec = {
        'alg': 'DPSampleOblGreedy',
        'k': k,
        'lambda_param': lambda_param,
        'eps': eps,
        'private': private,
        'selected': dp_sample_greedy_ob_selected,
        'value': dp_sample_ob_greedy_value,
        'queries': dp_sample_ob_greedy_queries
    }

    results.loc[len(results)] = rec

    ##################################################  Run DP Non-Oblivious Sample Greedy ##########################
    dp_sample_greedy_selected, dp_sample_greedy_value, dp_sample_greedy_queries = DP_sample_greedy(objective, ground_set, k, eps, private, False, gamma)
    print(f"DP Non-Oblivious Sample Greedy Objective Value: {dp_sample_greedy_value:.4f}")

    rec = {
        'alg': 'DPSampleGreedy',
        'k': k,
        'lambda_param': lambda_param,
        'eps': eps,
        'private': private,
        'selected': dp_sample_greedy_selected,
        'value': dp_sample_greedy_value,
        'queries': dp_sample_greedy_queries
    }

    results.loc[len(results)] = rec

    ##################################################  Run Random Baseline ######################################
    random_selected, random_value = random_baseline(objective, ground_set, k)
    print(f"Random Objective Value: {random_value:.4f}")

    rec = {
        'alg': 'Random',
        'k': k,
        'lambda_param': lambda_param,
        'eps': eps,
        'private': private,
        'selected': random_selected,
        'value': random_value,
        'queries': 0
    }

    results.loc[len(results)] = rec

    ##################################################  Save Results ##########################
    results.to_csv(f'Uber_results_{n_locs}_{spurious}.csv', index=False)

    # Visualize
    selected_hubs = grid_coords[dp_greedy_selected]

    # 1. Center the map on Manhattan
    m = folium.Map(location=[40.78, -73.97], zoom_start=12, tiles='CartoDB positron')

    # 2. Create Feature Groups (allows toggling them on/off)
    group_2 = folium.FeatureGroup(name='All').add_to(m)
    group_1 = folium.FeatureGroup(name='Non-Private Greedy').add_to(m)
    group_3 = folium.FeatureGroup(name='DP Greedy').add_to(m)
    group_4 = folium.FeatureGroup(name='Candidates').add_to(m)

    # 4. Add points from the second grid (e.g., your filtered or "on-land" grid)
    # Replace 'grid_2' with the variable name of your second coordinate array
    for lat, lon in passenger_coords:
        folium.CircleMarker(
            [lat, lon],
            radius=0.5,
            color='blue',
            fill=True,
        ).add_to(group_2)

    for lat, lon in grid_coords[nonpriv_selected]:
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color='red',
            fill=True,
        ).add_to(group_1)

    for lat, lon in grid_coords[dp_greedy_selected]:
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color='brown',
            fill=True,
        ).add_to(group_3)

    for lat, lon in grid_coords:
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color='yellow',
            fill=True,
        ).add_to(group_4)

    # 5. Add a layer control panel to the top-right
    folium.LayerControl().add_to(m)

    # Save the map
    m.save("Uber_results.html")
    return selected_hubs


# --- Execution ---
if __name__ == "__main__":

    full_island_hull = [
        (40.7005038, -74.0144209), (40.7112088, -73.9776851),
        (40.7282434, -73.9720702), (40.7418214, -73.9733576),
        (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482),
        (40.8769142, -73.9269985), (40.8512745, -73.9448513),
        (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    data_path = "C:\\Users\Ronza\Dev\DP-MSD\\Uber\\tmp.csv"

    k = 5
    eps_target = 0.1
    lambda_param = 0.1
    eps = 0.1
    private = True
    gamma = 0.1
    n_locs = 1000
    spurious = 500
    # params = {
    #     'k': 4,
    #     'lambda_param': 0.1,
    #     'eps': 0.1,
    #     'private': True,
    #     'gamma': 0.1
    # }

    # Initialize our pre-processor
    opt = UberOptimizer(full_island_hull, n_data=np.inf)

    # 2. Pre-process Data
    passenger_coords = opt.process_raw_data(data_path, "sampled_passengers.csv")

    grid_coords = opt.create_grid(n_locs=n_locs, spurious=spurious)
    # hubs_coords = pd.DataFrame(passenger_coords).sample(500)
    # hubs_coords.to_csv('hubs_coords.csv', index=False)
    # hubs_coords = pd.read_csv("hubs_coords.csv")
    # grid_coords = hubs_coords.values
    print(len(grid_coords))

    # 3. Setup Objective and GroundSet
    objective = MSDUberObjective(
        passenger_coords=passenger_coords,
        grid_coords=grid_coords,
        lambda_param=lambda_param,
        k=k,
        distortion=1.0,
        sensitivity=1 / len(passenger_coords)
    )

    # GroundSet is just the list of indices available in our grid
    ground_set = GroundSet(elements=list(range(len(grid_coords))))

    columns = ['alg', 'k', 'lambda_param', 'eps', 'private', 'selected', 'value', 'queries']
    results = pd.DataFrame(columns=columns)

    hubs = run_uber_experiment(k, lambda_param, eps, private, gamma, n_locs, spurious)
