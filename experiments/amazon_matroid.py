import os
import platform
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Set, Tuple

# Custom module imports
from src.algorithms import (
    local_search,
    DP_sample_local_search,
    random_baseline,
)
from src.classes import GroundSet, MSDAmazonObjectiveMat


def precompute_distances(meta_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    all_asins = meta_df['parent_asin'].unique()
    # Pre-convert to sets to avoid redundant processing in nested loops
    cat_lookup = {
        row.parent_asin: set(row.categories) if isinstance(row.categories, list) else set()
        for row in meta_df.itertuples()
    }

    dist_matrix = {asin: {} for asin in all_asins}

    for i, asin1 in enumerate(all_asins):
        set1 = cat_lookup.get(asin1, set())
        for j in range(i + 1, len(all_asins)):
            asin2 = all_asins[j]
            set2 = cat_lookup.get(asin2, set())

            intersection_len = len(set1 & set2)
            union_len = len(set1 | set2)

            dist = 1.0 - (intersection_len / union_len) if union_len > 0 else 1.0

            # Symmetric storage for O(1) lookup
            dist_matrix[asin1][asin2] = dist
            dist_matrix[asin2][asin1] = dist

    return dist_matrix


def run_matroid_experiment(
        objective: MSDAmazonObjectiveMat,
        ground_set: GroundSet,
        partition_map: Dict[str, str],
        partition_limits: Dict[str, int],
        params: Dict[str, Any],
        reps: int
) -> List[Dict[str, Any]]:
    results = []
    rk, eps, lam, gamma = params['k'], params['eps'], params['lambda'], params['gamma']
    is_private = params['private']

    # Define algorithm suite
    algorithms = [
        ('Random-Baseline', random_baseline, [objective, ground_set, partition_map, partition_limits, rk]),
        ('Local-Search-Matroid', local_search, [objective, ground_set, partition_map, partition_limits, rk, gamma]),
        ('DP-Sample-Local-Search', DP_sample_local_search,
         [objective, ground_set, partition_map, partition_limits, rk, eps, gamma, is_private]),
    ]

    for r in range(reps):
        print(f"  [Repetition {r + 1}/{reps}]")
        for name, alg_func, args in algorithms:
            start_ts = time.time()
            res = alg_func(*args)
            duration = time.time() - start_ts

            selected, value, rel, div = res[0], res[1], res[2], res[3]
            queries = res[4]

            results.append({
                'alg': name,
                'k': rk,
                'eps': eps,
                'lambda': lam,
                'gamma': gamma,
                'rep': r,
                'obj_value': value,
                'relevance': rel,
                'diversity': div,
                'queries': queries,
                'runtime_sec': round(duration, 4)
            })

    # Persistence
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results)
    output_path = "results/Amazon_Matroid_Results.csv"

    header_needed = not os.path.isfile(output_path)
    df_results.to_csv(output_path, mode='a', index=False, header=header_needed)

    return results


if __name__ == "__main__":
    # 1. Path and OS Setup
    prefix = '../' if platform.system() == 'Windows' else ''
    REVIEWS_PATH = f"{prefix}datasets/amazon/FULL_Health_and_Household.csv"
    META_PATH = f"{prefix}datasets/amazon/FULL_meta_Health_and_Household.csv"

    # 2. Data Loading & Filtering
    print("Ingesting Amazon Health and Household data...")
    reviews_df = pd.read_csv(REVIEWS_PATH, header=None, names=['user_id', 'parent_asin', 'rating', 'timestamp'])
    reviews_df['rating'] = 1

    meta_df = pd.read_csv(META_PATH, sep='\x1f', low_memory=False)
    meta_df = meta_df[meta_df['categories'].apply(lambda c: 'Health Care' in str(c))]
    meta_df = meta_df.sort_values(by='rating_number', ascending=False).head(1000)

    target_asins = meta_df['parent_asin'].unique()
    reviews_df = reviews_df[reviews_df['parent_asin'].isin(target_asins)]

    print(f"Filtered Dataset: {len(reviews_df)} reviews across {reviews_df['user_id'].nunique()} users.")

    partition_map = meta_df.set_index('parent_asin')['price_bin'].fillna('Unknown').to_dict()
    unique_bins = list(set(partition_map.values()))

    dist_matrix = precompute_distances(meta_df)
    g_set = GroundSet(elements=list(target_asins))

    cat_lookup = (
        meta_df.set_index('parent_asin')['categories']
        .astype(str).str.lower().str.split().apply(set).to_dict()
    )

    obj = MSDAmazonObjectiveMat(
        reviews_df=reviews_df,
        product_categories=cat_lookup,
        lambda_param=0.15,
        k=20,
        distortion=1.0,
        distance_matrix=dist_matrix
    )

    k_sweep = [
        {'k': k, 'eps': 0.2, 'lambda': 0.1, 'private': True, 'gamma': 0.1}
        for k in [14, 16, 18, 20]
    ]
    eps_sweep = [
        {'k': 6, 'eps': round(e, 2), 'lambda': 0.1, 'private': True, 'gamma': 0.1}
        for e in [0.12, 0.14, 0.16, 0.18, 0.20]
    ]
    lambda_sweep = [
        {'k': 6, 'eps': 0.2, 'lambda': l, 'private': True, 'gamma': 0.1}
        for l in [0.0, 0.2, 0.4, 0.6, 0.8]
    ]

    param_grid = k_sweep + eps_sweep + lambda_sweep

    for config in param_grid:
        bin_limit = int(np.ceil(config['k'] / len(unique_bins)))
        partition_limits = {p_bin: bin_limit for p_bin in unique_bins}

        print(f"\n{'=' * 30} CONFIG: k={config['k']}, eps={config['eps']}, lam={config['lambda']} {'=' * 30}")

        obj.lambda_param = config['lambda']
        obj.set_k(config['k'])

        run_matroid_experiment(
            objective=obj,
            ground_set=g_set,
            partition_map=partition_map,
            partition_limits=partition_limits,
            params=config,
            reps=10
        )