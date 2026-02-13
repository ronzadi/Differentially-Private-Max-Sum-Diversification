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

    df_results = pd.DataFrame(results)
    output_file = "results/Amazon_Greedy_Results.csv"
    df_results.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))

    return final_selected.get('DPGreedy')


# --- Execution ---
if __name__ == "__main__":
    reviews_path = "../datasets/amazon/FULL_Health_and_Household_Top10k_Dense.csv"
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    # reviews_df['rating'] = reviews_df['rating']/5.0
    reviews_df['rating'] = 1

    meta_path = "../datasets/amazon/FULL_meta_Health_and_Household_top10k.csv"
    meta_df = pd.read_csv(meta_path, sep='\x1f', low_memory=False)
    meta_df = meta_df[meta_df['categories'].apply(lambda c: 'Health Care' in c)]
    meta_df = meta_df.sort_values(by='rating_number', ascending=False).head(1000)

    # 1. Get the list of ASINs from the filtered meta_df
    selected_asins = meta_df['parent_asin'].unique()

    # 2. Filter reviews_df to only include those ASINs
    reviews_df = reviews_df[reviews_df['parent_asin'].isin(selected_asins)]
    print('num reviews: ', len(reviews_df))
    print('num users: ', len(reviews_df['user_id'].unique()))

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

    obj = MSDAmazonObjective(
        reviews_df=reviews_df,
        product_categories=product_categories_dict,
        lambda_param=0.15,
        k=20,
        distortion=1.0,
    )

    param_grid = [
        # {'k': 5, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 20, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 10, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 40, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 15, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 60, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 25, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 80, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 100, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k': 30, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        ###
        # {'k': 10, 'eps': 0.01, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.02, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.03, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.04, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.05, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.06, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.07, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.08, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.09, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.2, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.4, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.6, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.8, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 1, 'lambda': 0.15, 'private': True, 'gamma': 0.1},

        # {'k': 10, 'eps': 0.01, 'lambda': 0.15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.1, 'lambda': 0.1, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.03, 'lambda': .15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.1, 'lambda': 0.3, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.05, 'lambda': .15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.1, 'lambda': 0.5, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.07, 'lambda': .15, 'private': True, 'gamma': 0.1},
        {'k': 15, 'eps': 0.1, 'lambda': 0.7, 'private': True, 'gamma': 0.1},
        # {'k'2 10, 'eps': 0.09, 'lambda': .15, 'private': True, 'gamma': 0.1},
    ]

    for config in param_grid:
        print(
            f"\n================ CONFIG: k={config['k']}, eps={config['eps']}, lam={config['lambda']} ================")

        # Initialize Objective with processed data
        # obj = MSDAmazonObjective(
        #     reviews_df=reviews_df,
        #     product_categories=product_categories_dict,
        #     lambda_param=config['lambda'],
        #     k=config['k'],
        #     distortion=1.0,
        # )
        obj.lambda_param = config['lambda']
        obj.set_k(config['k'])

        run_amazon_experiment(obj, g_set, config, rep=10)  # Reduced reps for speed


