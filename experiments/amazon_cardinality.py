import math
import os
import platform
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Custom module imports
from experiments.amazon_matroid import precompute_distances
from src.classes import GroundSet, MSDAmazonObjective
from src.algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline
from src.dp_mechanisms import get_best_eps_0


def run_amazon_experiment(
        objective: MSDAmazonObjective,
        ground_set: GroundSet,
        params: Dict[str, Any],
        reps: int
) -> List[str]:
    results = []
    rk, eps = params['k'], params['eps']
    is_private, lam, gamma = params['private'], params['lambda'], params['gamma']

    # delta = 1/n^1.5 is standard for DP to ensure privacy holds across users
    delta_target = 1 / (objective.num_users ** 1.5)
    eps_0 = get_best_eps_0(eps_target=eps, delta_target=delta_target, k=rk)

    algorithms = [
        ('Non-Private', DP_greedy, [objective, ground_set, rk, eps_0, False]),
        ('DP-Greedy', DP_greedy, [objective, ground_set, rk, eps_0, is_private]),
        ('DP-Sample-Oblivious', DP_sample_greedy, [objective, ground_set, rk, eps_0, is_private, True, gamma]),
        ('DP-Sample-Greedy', DP_sample_greedy, [objective, ground_set, rk, eps_0, is_private, False, gamma]),
        ('Random', random_baseline, [objective, ground_set, rk])
    ]

    final_selected_asins = {}

    for r in range(reps):
        print(f"\n--- Iteration {r + 1}/{reps} ---")
        for name, alg_func, args in algorithms:
            print(f"  [Running] {name}")

            start_time = time.time()
            res = alg_func(*args)
            duration = time.time() - start_time

            selected, value, rel, div = res[0], res[1], res[2], res[3]
            queries = res[4] if len(res) > 4 else 0

            results.append({
                'alg': name,
                'k': rk,
                'lambda': lam,
                'eps_total': eps,
                'rep': r,
                'obj_value': value,
                'relevance': rel,
                'diversity': div,
                'queries': queries,
                'time_sec': round(duration, 4)
            })
            final_selected_asins[name] = selected

    # Save to CSV using append mode
    df_results = pd.DataFrame(results)
    output_path = "results/Amazon_Greedy_Results.csv"
    os.makedirs('results', exist_ok=True)

    df_results.to_csv(output_path, mode='a', index=False, header=not os.path.isfile(output_path))
    return final_selected_asins.get('DPGreedy')


if __name__ == "__main__":
    # 1. Configuration and Paths
    os_prefix = '../' if platform.system() == 'Windows' else ''
    REVIEWS_PATH = f"{os_prefix}datasets/amazon/FULL_Health_and_Household.csv"
    META_PATH = f"{os_prefix}datasets/amazon/FULL_meta_Health_and_Household.csv"

    # 2. Data Loading & Preprocessing
    print("Loading datasets...")
    reviews_df = pd.read_csv(REVIEWS_PATH, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating'] = 1  # Binary relevance setup

    meta_df = pd.read_csv(META_PATH, sep='\x1f', low_memory=False)
    # Filter for 'Health Care' category and pick top 1000 items by rating count
    meta_df = meta_df[meta_df['categories'].apply(lambda c: 'Health Care' in str(c))]
    meta_df = meta_df.sort_values(by='rating_number', ascending=False).head(1000)

    # Cross-reference ASINs between reviews and metadata
    target_asins = meta_df['parent_asin'].unique()
    reviews_df = reviews_df[reviews_df['parent_asin'].isin(target_asins)]

    print(f"Dataset Stats: {len(reviews_df)} reviews, {reviews_df['user_id'].nunique()} unique users.")

    # 3. Objective Component Setup
    print("Building category lookup and distance matrix...")
    product_categories = (
        meta_df.set_index('parent_asin')['categories']
        .astype(str).str.lower().str.split().apply(set).to_dict()
    )
    dist_matrix = precompute_distances(meta_df)
    ground_set = GroundSet(elements=list(target_asins))

    # 4. Define Experimental Sweeps (Including your Lambda sweep)
    # k-Sweep
    k_sweep = [
        {'k': k, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1}
        for k in [20, 40, 60, 80, 100]
    ]
    # Epsilon Sweep
    eps_sweep = [
        {'k': 60, 'eps': e, 'lambda': 0.1, 'private': True, 'gamma': 0.1}
        for e in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2]
    ]
    # Lambda Sweep (Restored from your comments)
    lambda_sweep = [
        {'k': 60, 'eps': 0.1, 'lambda': l, 'private': True, 'gamma': 0.1}
        for l in [0.0, 0.2, 0.4, 0.6, 0.8]
    ]

    param_grid = k_sweep + eps_sweep + lambda_sweep

    # 5. Execution Loop
    # Initialize objective once, update parameters inside loop for efficiency
    obj = MSDAmazonObjective(
        reviews_df=reviews_df,
        product_categories=product_categories,
        lambda_param=0.15,  # Placeholder
        k=20,  # Placeholder
        distortion=1.0,
        distance_matrix=dist_matrix
    )

    for config in param_grid:
        print(f"\n{'=' * 20} CONFIG: k={config['k']}, eps={config['eps']}, lam={config['lambda']} {'=' * 20}")

        # Dynamic update of objective parameters
        obj.lambda_param = config['lambda']
        obj.set_k(config['k'])

        run_amazon_experiment(obj, ground_set, config, reps=10)