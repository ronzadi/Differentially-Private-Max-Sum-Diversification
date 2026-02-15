import os
import random
import time
import math

import numpy as np
import pandas as pd

from dp_mechanisms import get_best_eps_0
from local_search_algorithms import (
    local_search,
    DP_sample_local_search,
    calculate_iterations,
     random_baseline,
)
from classes import GroundSet, MSDAmazonObjective
from greedy_algorithms import greedy


def precompute_distances(meta_df):
    """
    Pre-calculates all pairwise Jaccard distances for the ground set
    using meta_df for categories and reviews_df for the ASIN list.
    """
    # 1. Get unique ASINs from the reviews we are actually using
    all_asins = meta_df['parent_asin'].unique()

    # 2. Extract categories from meta_df as a lookup dict
    # Assuming meta_df has 'parent_asin' and 'categories' columns
    categories_lookup = dict(zip(meta_df['parent_asin'], meta_df['categories']))

    dist_matrix = {}

    for i, asin1 in enumerate(all_asins):
        # We handle the list-to-set conversion here to ensure intersection works
        set1 = set(categories_lookup.get(asin1, []))
        if asin1 not in dist_matrix:
            dist_matrix[asin1] = {}

        for j in range(i + 1, len(all_asins)):
            asin2 = all_asins[j]
            set2 = set(categories_lookup.get(asin2, []))

            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            dist = 1.0 - (intersection / union) if union > 0 else 1.0

            # Store bi-directionally for O(1) lookup
            dist_matrix[asin1][asin2] = dist
            if asin2 not in dist_matrix:
                dist_matrix[asin2] = {}
            dist_matrix[asin2][asin1] = dist

    return dist_matrix


def precompute_feasible_pair_values(ground_set:GroundSet, objective: MSDAmazonObjective):
    elements = ground_set.elements
    feasible_pair_values = {}

    # 1. Collect values for all feasible pairs
    for i, e1 in enumerate(elements):
        p1 = partition_map[e1]
        if partition_limits[p1] < 1:
            continue

        for e2 in elements[i + 1:]:
            p2 = partition_map[e2]
            if (p1 == p2 and partition_limits[p1] >= 2) or (p1 != p2 and partition_limits[p2] >= 1):
                val = objective.evaluate([e1, e2], distort=False)[0]
                feasible_pair_values[(e1, e2)] = val

    return feasible_pair_values

def run_matroid_experiment(objective, ground_set, partition_map, partition_limits, params, rep):
    """
    Executes a benchmark comparing non-private and private Local Search variants
    under partition matroid constraints.
    """
    results = []
    k = params['k']
    eps = params['eps']
    p = params['private']
    lam = params['lambda']
    g = params['gamma']

    # Algorithm Suite Definition
    algorithms = [
        ('RandomBaseline', random_baseline,
         [objective, ground_set, partition_map, partition_limits, k]),
        ('LocalSearch_Matroid', local_search,
         [objective, ground_set, partition_map, partition_limits, k, g]),
        ('DPSampleLocalSearch', DP_sample_local_search,
         [objective, ground_set, partition_map, partition_limits, k, eps, g, p]),
    ]

    for i in range(rep):
        print(f"\n--- Repetition {i + 1}/{rep} ---")
        for name, func, args in algorithms:
            if i > 0 and name == 'LocalSearch_Matroid':
                continue
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
                'time_sec': round(duration, 4),
                'timestamp': round(time.time() * 1000)
            })

    # Data Persistence
    df_results = pd.DataFrame(results)
    output_file = "results/Amazon_Matroid_Results8-14.csv"

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    df_results.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))

    return results


if __name__ == "__main__":
    # --- 1. Data Ingestion & Preprocessing ---
    reviews_path = "../datasets/amazon/FULL_Health_and_Household_Top10k_Dense.csv"
    reviews_df = pd.read_csv(reviews_path, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])

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

    # Optional, sample users
    ##################
    # sampled_users = random.sample(list(reviews_df['user_id'].unique()), 100000)
    # reviews_df = reviews_df[reviews_df['user_id'].isin(sampled_users)]
    ###################
    print('num users: ', len(reviews_df['user_id'].unique()))

    # --- 2. Constraint Modeling (Partition Matroid) ---
    # Constraint: One item per 'price_bin'
    partition_map = meta_df.set_index('parent_asin')['price_bin'].fillna('Unknown').to_dict()
    unique_price_bins = meta_df['price_bin'].fillna('Unknown').unique()

    # --- 3. Objective & Ground Set Initialization ---
    all_asins = list(meta_df['parent_asin'].unique())
    g_set = GroundSet(elements=all_asins)

    # Get pairwise distances matrix
    distance_matrix = precompute_distances(meta_df)

    # Log dataset statistics
    print(f'Total Reviews: {len(reviews_df)}')
    print(f'Unique Users:  {len(reviews_df["user_id"].unique())}')

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
        distance_matrix=distance_matrix
    )

    # feasible_pair_values = precompute_feasible_pair_values(g_set, obj)

    # --- 4. Experimental Parameter Grid ---
    param_grid = [
        # Impact of Cardinality k
        {'k': 4, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 8, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 10, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 12, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 14, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},

        # Privacy Budget (Epsilon) Sensitivity
        # {'k': 6, 'eps': 0.02, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        # {'k': 6, 'eps': 0.04, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        # {'k': 6, 'eps': 0.06, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        # {'k': 6, 'eps': 0.08, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.1, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.3, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.4, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.6, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.7, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.8, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.9, 'lambda': 0.1, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 1.0, 'lambda': 0.1, 'private': True, 'gamma': 0.2},

        # Diversity Weight (Lambda) Sensitivity
        {'k': 6, 'eps': 0.5, 'lambda': 0, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.2, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.4, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.6, 'private': True, 'gamma': 0.2},
        {'k': 6, 'eps': 0.5, 'lambda': 0.8, 'private': True, 'gamma': 0.2},
    ]

    # --- 5. Execution Loop ---
    for config in param_grid:

        limit = int(np.ceil(config['k'] / len(unique_price_bins)))
        partition_limits = {price_bin: limit for price_bin in unique_price_bins}

        # Feasibility check against matroid constraints
        print(f"\n{'=' * 80}")
        print(f"CONFIG: k={config['k']}, eps={config['eps']}, lambda={config['lambda']}")
        print(f"{'=' * 80}")

        obj.lambda_param = config['lambda']
        obj.set_k(config['k'])

        run_matroid_experiment(
            obj,
            g_set,
            partition_map,
            partition_limits,
            config,
            rep=5
        )