import random

import numpy as np

from classes import MSDObjective, GroundSet
from dp_mechanisms import exp_mech


def get_initial_set(objective, ground_set, partition_map, partition_limits, k):
    elements = ground_set.elements
    best_pair, max_val = [], -float('inf')

    # 1. Find the best feasible pair
    for i, e1 in enumerate(elements):
        p1 = partition_map[e1]
        if partition_limits[p1] < 1: continue

        for e2 in elements[i + 1:]:
            p2 = partition_map[e2]
            # Check feasibility: if same partition, need limit >= 2; else both need >= 1
            if (p1 == p2 and partition_limits[p1] >= 2) or (p1 != p2 and partition_limits[p2] >= 1):
                val = objective.evaluate([e1, e2], distort=False)[0]
                if val > max_val:
                    max_val, best_pair = val, [e1, e2]

    # 2. Complete to base arbitrarily
    S = list(best_pair)
    counts = {p: 0 for p in partition_limits}
    for e in S: counts[partition_map[e]] += 1

    for e in elements:
        if len(S) >= k: break
        p = partition_map[e]
        if e not in S and counts[p] < partition_limits[p]:
            S.append(e)
            counts[p] += 1

    return S


def local_search(objective, ground_set, partition_map, partition_limits, k, gamma):
    # Step 0: Custom Initialization
    initial_S = get_initial_set(objective, ground_set, partition_map, partition_limits, k)
    S = list(initial_S)

    # Initial evaluation
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)

    while True:
        swap_values = {}
        S_set = set(S)

        partition_counts = {p: 0 for p in partition_limits}
        for e in S:
            p_id = partition_map[e]
            partition_counts[p_id] += 1

        # 1. Build dictionary of feasible swaps
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in ground_set.elements:
                if e_in in S_set:
                    continue

                p_in_id = partition_map[e_in]

                # Feasibility check
                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) + 1 <= partition_limits[p_in_id]:
                    idx = S.index(e_out)
                    S_tmp = S[:idx] + S[idx + 1:] + [e_in]
                    val, _, _, _ = objective.evaluate(S_tmp, distort=False)
                    swap_values[(e_out, e_in)] = val

        if not swap_values:
            break

        # 2. Select best swap
        best_swap = max(swap_values, key=swap_values.get)
        best_val = swap_values[best_swap]

        # 3. Test threshold and commit
        if best_val > current_val * (1 + gamma / k):
            e_out, e_in = best_swap
            idx = S.index(e_out)
            S_tmp = S[:idx] + S[idx + 1:] + [e_in]

            # Final state re-computation
            current_val, coverage, dist, auxiliary = objective.evaluate(S_tmp, distort=False)
            S = S_tmp
            print(f"Committed: {e_out} -> {e_in} | New Val: {current_val:.4f}")
        else:
            break
    print(f"Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
    return S, current_val, coverage, dist, objective.num_queries








def random_baseline(objective: MSDObjective, ground_set: GroundSet, k):
    pass