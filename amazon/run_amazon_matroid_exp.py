import os
import time
import pandas as pd
from local_search_algorithms import local_search  # Your updated LS
from classes import GroundSet, MSDAmazonObjective
from greedy_algorithms import greedy


def run_matroid_experiment(objective, ground_set, partition_map, partition_limits, params, rep):
    results = []
    k, lam, gamma = params['k'], params['lambda'], params['gamma']

    # We compare the standard Greedy (which might violate store constraints)
    # vs. the Local Search (which respects the intersection)
    algorithms = [
        ('LocalSearch_Matroid', local_search, [objective, ground_set, partition_map, partition_limits, k, gamma])
    ]

    for i in range(rep):
        print(f"\n--- Repetition {i + 1}/{rep} ---")
        for name, func, args in algorithms:
            start_time = time.time()
            res = func(*args)
            duration = time.time() - start_time

            selected, value, rel, div = res[0], res[1], res[2], res[3]
            queries = res[4]

            results.append({
                'alg': name,
                'k': k,
                'lambda_param': lam,
                'gamma': gamma,
                'rep': i,
                'value': value,
                'relevance': rel,
                'diversity': div,
                'queries': queries,
                'time_sec': round(duration, 4)
            })

    df_results = pd.DataFrame(results)
    output_file = "results/Amazon_Matroid_Results.csv"
    df_results.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))
    return selected


if __name__ == "__main__":
    # --- 1. Load Data ---
    reviews_path = "../datasets/amazon/FULL_Health_and_Household_Top10k_Dense.csv"
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating'] = reviews_df['rating'] / 5.0

    meta_path = "../datasets/amazon/FULL_meta_Health_and_Household_top10k.csv"
    meta_df = pd.read_csv(meta_path, sep='\x1f', low_memory=False).sort_values(by='rating_number',
                                                                               ascending=False).head(100)

    # --- 2. Setup the "One Item Per Store" Partition Matroid ---
    # We treat each store as a partition.
    partition_map = meta_df.set_index('parent_asin')['store'].fillna('Unknown').to_dict()

    # Constraint: At most 1 item per store
    unique_stores = meta_df['store'].fillna('Unknown').unique()
    partition_limits = {store: 1 for store in unique_stores}

    # --- 3. Ground Set & Objective ---
    all_asins = list(meta_df['parent_asin'].unique())
    g_set = GroundSet(elements=all_asins)

    print("Preparing category lookup...")
    product_categories_dict = (
        meta_df.set_index('parent_asin')['categories']
        .astype(str)
        .str.lower()
        .str.split()
        .apply(set)
        .to_dict()
    )

    obj = MSDAmazonObjective(
        reviews_df=reviews_df,
        product_categories=product_categories_dict,
        lambda_param=0.15,
        k=20,
        distortion=0,
    )

    # --- 4. Execution Loop ---
    param_grid = [
        {'k': 5, 'lambda': 0.15, 'gamma': 0.1},
        {'k': 15, 'lambda': 0.15, 'gamma': 0.1},
        {'k': 25, 'lambda': 0.15, 'gamma': 0.1},
        {'k': 35, 'lambda': 0.15, 'gamma': 0.1},
    ]

    for config in param_grid:
        # Check if k is feasible (k cannot exceed number of unique stores)
        if config['k'] > len(unique_stores):
            print(f"Skipping k={config['k']}: Not enough unique stores.")
            continue

        print(f"\n=== Running: k={config['k']}, gamma={config['gamma']} ===")
        obj.set_k(config['k'])
        obj.lambda_param = config['lambda']

        run_matroid_experiment(obj, g_set, partition_map, partition_limits, config, rep=1)