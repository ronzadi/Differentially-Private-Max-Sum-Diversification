import math
import os
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
            print(f"Running {name}...")
            res = func(*args)
            selected, value = res[0], res[1]
            queries = res[2] if len(res) > 2 else 0

            results.append({
                'alg': name, 'k': k, 'lambda_param': lam, 'eps': eps,
                'rep': i, 'value': value, 'queries': queries
            })
            final_selected[name] = selected

    df_results = pd.DataFrame(results)
    output_file = "Amazon_Master_Results.csv"
    df_results.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))

    return final_selected.get('DPGreedy')


# --- Execution ---
if __name__ == "__main__":
    # 1. Load Reviews (1% sample)
    reviews_path = "../datasets/amazon/Health_and_Household_1_percent.csv"
    # Assuming columns are: user_id, parent_asin, rating, timestamp
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating']= reviews_df['rating']/5.0

    # 2. Load Metadata (The stable file we created)
    meta_path = "../datasets/amazon/meta_Health_and_Household.csv"
    meta_df = pd.read_csv(meta_path, sep='\x1f', low_memory=False)

    # 3. Pre-process Categories into a Dictionary of Sets {asin: {cat1, cat2}}
    # This is crucial for the Jaccard distance logic
    print("Preparing category lookup...")
    product_categories_dict = {}
    for _, row in meta_df.iterrows():
        cat_str = str(row['categories'])
        # Clean the string and turn into a set of words/tags
        product_categories_dict[row['parent_asin']] = set(cat_str.lower().split())

    # 4. Define the Ground Set (The actual product IDs available to pick from)
    all_asins = list(meta_df['parent_asin'].unique())
    g_set = GroundSet(elements=all_asins)

    param_grid = [
        {'k': 3, 'eps': 0.1, 'lambda': 0.2, 'private': False, 'gamma': 0.1},
        # {'k': 8, 'eps': 0.1, 'lambda': 0.2, 'private': True, 'gamma': 0.1},
        # {'k': 8, 'eps': 0.4, 'lambda': 0.2, 'private': True, 'gamma': 0.1},
        # {'k': 8, 'eps': 1.0, 'lambda': 0.2, 'private': True, 'gamma': 0.1},
        # {'k': 8, 'eps': 0.1, 'lambda': 0.8, 'private': True, 'gamma': 0.1},
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

        run_amazon_experiment(obj, g_set, config, rep=3)  # Reduced reps for speed