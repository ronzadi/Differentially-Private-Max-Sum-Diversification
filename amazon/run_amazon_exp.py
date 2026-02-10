import math
import os
import time

import pandas as pd
import numpy as np
from classes import GroundSet, MSDAmazonObjective
from greedy_algorithms import greedy, DP_greedy, DP_sample_greedy, random_baseline
from dp_mechanisms import get_best_eps_0


def run_amazon_experiment(objective, ground_set, params, rep):
    results = []
    k, eps, p, lam, g = params['k'], params['eps'], params['private'], params['lambda'], params['gamma']

    # Sensitivity for Amazon Ratings is usually the max rating (e.g., 5.0) / num_users
    delta_target = 1 / (objective.num_users ** 1.5)
    eps_0 = get_best_eps_0(eps_target=eps, delta_target=delta_target, k=k)

    algorithms = [
        ('nonpriv', greedy, [objective, ground_set, k]),
        ('DPGreedy', DP_greedy, [objective, ground_set, k, eps_0, p]),
        ('DPSampleOblGreedy', DP_sample_greedy, [objective, ground_set, k, eps_0, p, True, g]),
        ('DPSampleGreedy', DP_sample_greedy, [objective, ground_set, k, eps_0, p, False, g]),
        ('Random', random_baseline, [objective, ground_set, k])
    ]

    final_selected = {}
    for i in range(rep):
        print(f"\n--- Repetition {i + 1}/{rep} ---")
        for name, func, args in algorithms:
            # --- Timing Start ---
            start_time = time.time()
            res = func(*args)
            end_time = time.time()
            duration = end_time - start_time
            # --- Timing End ---
            selected, value, rel, div = res[0], res[1], res[2], res[3]
            queries = res[2] if len(res) > 2 else 0

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

    df_results = pd.DataFrame(results)
    output_file = "results/Amazon_Master_Results.csv"
    df_results.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))

    return final_selected.get('DPGreedy')


# --- Execution ---
if __name__ == "__main__":
    reviews_path = "../datasets/amazon/reviews_Health_and_Household.csv"
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating'] = reviews_df['rating']/5.0

    meta_path = "../datasets/amazon/meta_Health_and_Household.csv"
    meta_df = pd.read_csv(meta_path, sep='\x1f', low_memory=False)

    print("Preparing category lookup...")
    product_categories_dict = (
        meta_df.set_index('parent_asin')['categories']
        .astype(str)
        .str.lower()
        .str.split()
        .apply(set)
        .to_dict()
    )

    # 4. Define the Ground Set (The actual product IDs available to pick from)
    all_asins = list(meta_df['parent_asin'].unique())
    g_set = GroundSet(elements=all_asins)

    param_grid = [
        {'k': 20, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k': 20, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 40, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k': 40, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 60, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k': 70, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 80, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k': 90, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 100, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        ###
        # {'k': 10, 'eps': 0.01, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.02, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.03, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.04, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.05, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.06, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.07, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.08, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.09, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.15, 'private': False, 'gamma': 0.1},

        # {'k': 10, 'eps': 0.01, 'lambda': 0.15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.2, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.03, 'lambda': .15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.4, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.05, 'lambda': .15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.6, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.07, 'lambda': .15, 'private': False, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.8, 'private': False, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.09, 'lambda': .15, 'private': False, 'gamma': 0.1},
    ]

    for config in param_grid:
        print(
            f"\n================ CONFIG: k={config['k']}, eps={config['eps']}, lam={config['lambda']} ================")

        # Initialize Objective with processed data
        obj = MSDAmazonObjective(
            reviews_df=reviews_df,
            product_categories=product_categories_dict,
            lambda_param=config['lambda'],
            k=config['k'],
            distortion=1.0,
        )

        run_amazon_experiment(obj, g_set, config, rep=1)  # Reduced reps for speed