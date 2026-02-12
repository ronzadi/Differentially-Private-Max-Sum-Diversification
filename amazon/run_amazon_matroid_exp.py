import os
import time
import pandas as pd

from dp_mechanisms import get_best_eps_0
from local_search_algorithms import local_search, DP_sample_local_search, calculate_iterations, \
    DP_sample_local_search_threshold
from classes import GroundSet, MSDAmazonObjective
from greedy_algorithms import greedy


def run_matroid_experiment(objective, ground_set, partition_map, partition_limits, params, rep):
    results = []
    k, eps, p, lam, g = params['k'], params['eps'], params['private'], params['lambda'], params['gamma']

    # Adding both the non-private and the new DP Sample LS to the comparison
    algorithms = [
        ('LocalSearch_Matroid', local_search, [objective, ground_set, partition_map, partition_limits, k, g]),
        ('DPSampleLocalSearch', DP_sample_local_search,
         [objective, ground_set, partition_map, partition_limits, k, eps, g, p]),
        ('DPSampleLocalSearchThreshold', DP_sample_local_search_threshold,
         [objective, ground_set, partition_map, partition_limits, k, eps, g, p])
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
                'eps': eps,
                'private': p,
                'lambda_param': lam,
                'gamma': g,
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
    return results


if __name__ == "__main__":
    # --- 1. Load Data ---
    # (Keeping your specific data paths)
    reviews_path = "../datasets/amazon/FULL_Health_and_Household_Top10k_Dense.csv"
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating'] = reviews_df['rating'] / 5.0

    meta_path = "../datasets/amazon/FULL_meta_Health_and_Household_top10k.csv"
    meta_df = pd.read_csv(meta_path, sep='\x1f', low_memory=False).sort_values(by='rating_number',
                                                                               ascending=False).head(200)

    # --- 2. Setup the "One Item Per Store" Partition Matroid ---
    partition_map = meta_df.set_index('parent_asin')['store'].fillna('Unknown').to_dict()
    unique_stores = meta_df['store'].fillna('Unknown').unique()
    partition_limits = {store: 1 for store in unique_stores}

    # --- 3. Ground Set & Objective ---
    all_asins = list(meta_df['parent_asin'].unique())
    # 2. Filter reviews_df to only include those ASINs
    # reviews_df = reviews_df[reviews_df['parent_asin'].isin(all_asins)]
    print('num reviews: ', len(reviews_df))
    print('num users: ', len(reviews_df['user_id'].unique()))

    g_set = GroundSet(elements=all_asins)

    product_categories_dict = (
        meta_df.set_index('parent_asin')['categories']
        .astype(str).str.lower().str.split().apply(set).to_dict()
    )

    obj = MSDAmazonObjective(
        reviews_df=reviews_df,
        product_categories=product_categories_dict,
        lambda_param=0.15,
        k=20,
        distortion=1.0,
    )

    # --- 4. Parameter Grid ---
    param_grid = [
        {'k': 4, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 8, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 12, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 14, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 16, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 18, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 20, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},

        {'k': 6, 'eps': 0.02, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.04, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.06, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.08, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.2, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.4, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.6, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.8, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 1.0, 'lambda': 0.15, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.3, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.5, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.7, 'private': True, 'gamma': 0.2},
    ]

    for config in param_grid:
        if config['k'] > len(unique_stores):
            print(f"Skipping k={config['k']}: Not enough unique stores.")
            continue

        print(
            f"\n================ CONFIG: k={config['k']}, eps={config['eps']}, lam={config['lambda']} ================")
        obj.lambda_param = config['lambda']
        obj.set_k(config['k'])

        run_matroid_experiment(obj, g_set, partition_map, partition_limits, config, rep=19)