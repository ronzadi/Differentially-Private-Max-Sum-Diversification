import numpy as np
import pandas as pd
import random
from prep import UberOptimizer
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline
import numpy as np
import pandas as pd
from classes import MSDFacilityLocation, GroundSet
import numpy as np
import pandas as pd


def run_uber_experiment(input_csv, k, lambda_param, eps, private, gamma):
    # manhattan_box = [40.81794, 40.6866, 40.80204, 40.71315,
    #                  -73.96483, -73.99197, -73.91436, -74.04519]
    full_island_hull = [
        (40.7005038, -74.0144209), (40.7112088, -73.9776851),
        (40.7282434, -73.9720702), (40.7418214, -73.9733576),
        (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482),
        (40.8769142, -73.9269985), (40.8512745, -73.9448513),
        (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    # Initialize our pre-processor
    opt = UberOptimizer(full_island_hull, n_data=np.inf)

    # 2. Pre-process Data
    passenger_coords = opt.process_raw_data(input_csv, "sampled_passengers.csv")

    # Using your optimized 2000-point, 80-column grid
    grid_coords = opt.create_grid(n_locs=50)
    # hubs_coords = pd.DataFrame(passenger_coords).sample(500)
    # hubs_coords.to_csv('hubs_coords.csv', index=False)
    # hubs_coords = pd.read_csv("hubs_coords.csv")
    # grid_coords = hubs_coords.values

    # 3. Setup Objective and GroundSet
    # Note: We pass the 'opt.norm' to the objective to ensure correct scaling
    objective = MSDFacilityLocation(
        passenger_coords=passenger_coords,
        grid_coords=grid_coords,
        lambda_param=lambda_param,
        k=k,
        distortion=1.0,  # Greedy will set this to 0.5 internally
        sensitivity=1/len(passenger_coords)
    )

    # GroundSet is just the list of indices available in our grid
    ground_set = GroundSet(elements=list(range(len(grid_coords))))

    #  Run Non-Private Greedy ##################################################
    nonpriv_selected, nonpriv_value = greedy(objective, ground_set, k)
    print(f"Non-private Greedy Objective Value: {nonpriv_value:.4f}")

    #  Run DP Greedy ##################################################
    dp_greedy_selected, dp_greedy_value = DP_greedy(objective, ground_set, k, eps, private)
    print(f"DP Greedy Objective Value: {dp_greedy_value:.4f}")

    #  Run DP Oblivious Sample Greedy ##################################################
    dp_greedy_selected, dp_greedy_value = DP_sample_greedy(objective, ground_set, k, eps, private, True, gamma)
    print(f"DP Oblivious Sample Greedy Objective Value: {dp_greedy_value:.4f}")

    #  Run DP Non-Oblivious Sample Greedy ##################################################
    dp_greedy_selected, dp_greedy_value = DP_sample_greedy(objective, ground_set, k, eps, private, True, gamma)
    print(f"DP Non-Oblivious Sample Greedy Objective Value: {dp_greedy_value:.4f}")

    random_selected, random_value = random_baseline(objective, ground_set, k)
    print(f"Random Objective Value: {random_value:.4f}")

    #  Export Results
    selected_hubs = grid_coords[dp_greedy_selected]
    results_df = pd.DataFrame(selected_hubs, columns=['lat', 'lon'])
    results_df.to_csv("Uber_results.csv", index=False)

    print(f"Hubs saved to 'manhattan_hubs.csv'")

    import folium

    # 1. Center the map on Manhattan
    m = folium.Map(location=[40.78, -73.97], zoom_start=12, tiles='CartoDB positron')

    # 2. Create Feature Groups (allows toggling them on/off)
    group_2 = folium.FeatureGroup(name='All').add_to(m)
    group_1 = folium.FeatureGroup(name='Non-Private Greedy').add_to(m)
    group_3 = folium.FeatureGroup(name='DP Greedy').add_to(m)

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

    # 5. Add a layer control panel to the top-right
    folium.LayerControl().add_to(m)

    # Save the map
    m.save("Uber_results.html")
    return selected_hubs


# --- Execution ---
if __name__ == "__main__":
    # hubs = run_manhattan_experiment("uber-raw-data-apr14.csv")
    # hubs = run_manhattan_experiment("C:\\Users\Ronza\Dev\DP-MSD\datasets\\uber-raw-data-jun14.csv")
    data_path = "C:\\Users\Ronza\Dev\DP-MSD\\Uber\\tmp.csv"

    k = 5
    eps_target = 0.1

    params = {
        'k': 10,
        'lambda_param': 0.1,
        'eps': 0.1,
        'private': True,
        'gamma': 0.1
    }

    hubs = run_uber_experiment(data_path, **params)