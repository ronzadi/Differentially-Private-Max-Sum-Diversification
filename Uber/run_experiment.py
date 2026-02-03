import numpy as np
import pandas as pd
import random
from prep import UberOptimizer
from algorithms import greedy
import numpy as np
import pandas as pd
from classes import MSDFacilityLocation, GroundSet
import numpy as np
import pandas as pd


def run_manhattan_experiment(input_csv, k=5, lambda_param=1):
    # 1. Setup Environment
    # Coordinates from your optimized box
    # manhattan_box = [40.81794, 40.6866, 40.80204, 40.71315,
    #                  -73.96483, -73.99197, -73.91436, -74.04519]
    manhattan_box = [
        40.88000,  # North (Inwood) - Fixes the "missing top" issue
        40.70000,  # South (The Battery)
        40.78500,  # East (Upper East Side anchor)
        40.75500,  # West (Hell's Kitchen anchor)
        -73.92500,  # lonNorth
        -74.01500,  # lonSouth
        -73.93500,  # lonEast - Narrowed to hug the East River
        -74.01000  # lonWest - Narrowed to hug the Hudson
    ]

    # Initialize our pre-processor
    opt = UberOptimizer(manhattan_box, n_data=20000)

    # 2. Pre-process Data
    print("Step 1: Filtering and Sampling Uber Data...")
    # This executes the reservoir sampling (10k) and tilted-box filter
    passenger_coords = opt.process_raw_data(input_csv, "sampled_passengers.csv")

    print("Step 2: Generating Candidate Grid (Land-Only)...")
    # Using your optimized 2000-point, 80-column grid
    grid_coords = opt.create_grid(n_locs=500, n_cols=40)
    # grid_coords = pd.DataFrame(passenger_coords).sample(500).values

    # 3. Setup Objective and GroundSet
    # Note: We pass the 'opt.norm' to the objective to ensure correct scaling
    objective = MSDFacilityLocation(
        passenger_coords=passenger_coords,
        grid_coords=grid_coords,
        lambda_param=lambda_param,
        k=k,
        distortion=1.0  # Greedy will set this to 0.5 internally
    )

    # GroundSet is just the list of indices available in our grid
    ground_set = GroundSet(elements=list(range(len(grid_coords))))

    # 4. Run Optimization
    print(f"Step 3: Running Greedy Selection for k={k}...")
    selected_indices, final_value = greedy(objective, ground_set, k)

    # 5. Export Results
    selected_hubs = grid_coords[selected_indices]
    results_df = pd.DataFrame(selected_hubs, columns=['lat', 'lon'])
    results_df.to_csv("Uber_results.csv", index=False)

    print(f"\nExperiment Finished!")
    print(f"Final Objective Value: {final_value:.4f}")
    print(f"Hubs saved to 'manhattan_hubs.csv'")

    import folium

    # 1. Center the map on Manhattan
    m = folium.Map(location=[40.78, -73.97], zoom_start=12, tiles='CartoDB positron')

    # 2. Create Feature Groups (allows toggling them on/off)
    group_1 = folium.FeatureGroup(name='Original Grid (Red)').add_to(m)
    group_2 = folium.FeatureGroup(name='Optimized Grid (Blue)').add_to(m)


    # 4. Add points from the second grid (e.g., your filtered or "on-land" grid)
    # Replace 'grid_2' with the variable name of your second coordinate array
    for lat, lon in passenger_coords:
        folium.CircleMarker(
            [lat, lon],
            radius=0.5,
            color='blue',
            fill=True,
        ).add_to(group_2)

    # 3. Add points from the first grid (e.g., your full candidate set)
    for lat, lon in selected_hubs:
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color='red',
            fill=True,
        ).add_to(group_1)

    # 5. Add a layer control panel to the top-right
    folium.LayerControl().add_to(m)

    # Save the map
    m.save("Uber_results.html")
    return selected_hubs


# --- Execution ---
if __name__ == "__main__":
    # hubs = run_manhattan_experiment("uber-raw-data-apr14.csv")
    # hubs = run_manhattan_experiment("C:\\Users\Ronza\Dev\DP-MSD\datasets\\uber-raw-data-jun14.csv")
    hubs = run_manhattan_experiment("C:\\Users\Ronza\Dev\DP-MSD\\Uber\\tmp.csv")


    # Visualize locations and Selected hubs
