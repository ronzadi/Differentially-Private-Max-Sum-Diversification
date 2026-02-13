import math
import random
import numpy as np

from classes import MSDObjective, GroundSet
from dp_mechanisms import exp_mech, get_best_eps_0


def calculate_iterations(k, gamma):
    """
    Calculates the iteration threshold T.
    Formula: T = ceil((2 * k * log(k)) / (gamma * (1 - 1/e)))
    """
    numerator = 2 * k * math.log(k)
    denominator = gamma * (1 - 1 / math.e)
    return math.ceil(numerator / denominator)


def get_arbitrary_feasible(ground_set, partition_map, partition_limits, k):
    """
    Finds an arbitrary feasible base (size k) for the matroid intersection.
    Used for basic initialization.
    """
    S = []
    counts = {p: 0 for p in partition_limits}

    for e in ground_set.elements:
        if len(S) == k:
            break
        p_id = partition_map.get(e)
        if p_id in partition_limits and counts[p_id] < partition_limits[p_id]:
            S.append(e)
            counts[p_id] += 1

    if len(S) < k:
        print(f"Warning: Could only find {len(S)} feasible elements.")
    return S


def get_initial_set(objective, ground_set, partition_map, partition_limits, k, eps=None, private=False):
    """
    Finds the best feasible pair (optionally privately) and completes it to size k.
    """
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

    # 2. Select the pair
    if private:
        if eps is None:
            raise ValueError("eps must be provided for private initialization.")
        best_pair_tuple = exp_mech(feasible_pair_values, eps, objective.sensitivity, private=True)
    else:
        best_pair_tuple = max(feasible_pair_values, key=feasible_pair_values.get)

    # 3. Complete to base arbitrarily
    S = list(best_pair_tuple)
    counts = {p: 0 for p in partition_limits}
    for e in S:
        counts[partition_map[e]] += 1

    for e in elements:
        if len(S) >= k:
            break
        p = partition_map[e]
        if e not in S and counts[p] < partition_limits[p]:
            S.append(e)
            counts[p] += 1

    return S


def local_search(objective, ground_set :GroundSet, partition_map, partition_limits, k, gamma):
    """
    Standard deterministic Local Search with threshold-based swaps.
    """
    S = get_initial_set(objective, ground_set, partition_map, partition_limits, k)
    # S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
    print('Got initial set', S)
    current_val, coverage, dist, _ = objective.evaluate(S, distort=False)

    partition_counts = {p: 0 for p in partition_limits}
    for e in S:
        partition_counts[partition_map[e]] += 1

    while True:
        swap_values = {}
        S_set = set(S)
        # partition_counts = {p: S.count(e) for e, p in partition_map.items() if
        #                     e in S_set}  # Optimization note: simplified for brevity

        # Recalculate partition counts properly
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in ground_set.elements:
                if e_in in S_set:
                    continue
                p_in_id = partition_map[e_in]

                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) + 1 <= partition_limits[p_in_id]:
                    idx = S.index(e_out)
                    S_tmp = S[:idx] + S[idx + 1:] + [e_in]
                    val, _, _, _ = objective.evaluate(S_tmp, distort=False)
                    swap_values[(e_out, e_in)] = val

        if not swap_values:
            break

        best_swap = max(swap_values, key=swap_values.get)
        best_val = swap_values[best_swap]

        if best_val > current_val * (1 + gamma / k):
            e_out, e_in = best_swap
            idx = S.index(e_out)
            S = S[:idx] + S[idx + 1:] + [e_in]
            # --- UPDATE COUNTS INSTEAD OF RECOMPUTING ---
            partition_counts[partition_map[e_out]] -= 1
            partition_counts[partition_map[e_in]] += 1
            current_val, coverage, dist, _ = objective.evaluate(S, distort=False)
            print(f"Committed: {e_out} -> {e_in} | New Val: {current_val:.4f}")
        else:
            break

    return S, current_val, coverage, dist, objective.num_queries


def DP_sample_local_search(objective, ground_set, partition_map, partition_limits, k, eps, gamma, private):
    """
    DP Local Search using sampling to reduce sensitivity and query count.
    Updates the set at every iteration for T iterations.
    """
    delta_target = 1 / (objective.num_users ** 1.5)
    T = calculate_iterations(k, gamma)
    eps_0 = get_best_eps_0(eps_target=eps/2, delta_target=delta_target, k=T, decomposable=False)

    S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
    print('Got initial set', S)

    observed_sets = {tuple(sorted(S)): objective.evaluate(S, distort=False)[0]}
    sample_size = math.ceil(ground_set.nb_elements / k)
    partition_counts = {p: 0 for p in partition_limits}
    for e in S:
        partition_counts[partition_map[e]] += 1

    for it in range(int(T)):
        S_set = set(S)
        feasible_swaps = []
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in ground_set.elements:
                if e_in in S_set: continue
                p_in_id = partition_map[e_in]
                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) < partition_limits[p_in_id]:
                    feasible_swaps.append((e_out, e_in))

        if not feasible_swaps:
            break

        sampled_keys = random.sample(feasible_swaps, min(len(feasible_swaps), sample_size))
        sampled_swap_values = {}
        for e_out, e_in in sampled_keys:
            idx = S.index(e_out)
            S_tmp = S[:idx] + S[idx + 1:] + [e_in]
            sampled_swap_values[(e_out, e_in)] = objective.evaluate(S_tmp, distort=False)[0]

        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        e_out, e_in = best_swap
        idx = S.index(e_out)
        S = S[:idx] + S[idx + 1:] + [e_in]
        # --- UPDATE COUNTS INSTEAD OF RECOMPUTING ---
        partition_counts[partition_map[e_out]] -= 1
        partition_counts[partition_map[e_in]] += 1

        current_val = objective.evaluate(S, distort=False)[0]
        observed_sets[tuple(sorted(S))] = current_val
        print('current_val:', current_val)

    S_best = list(exp_mech(observed_sets, eps/2, objective.sensitivity, private=private))
    val, cov, div, _ = objective.evaluate(S_best, distort=False)
    return S_best, val, cov, div, objective.num_queries


def DP_sample_local_search_threshold(objective, ground_set, partition_map, partition_limits, k, eps, gamma, private):
    """
    DP Local Search that uses a noisy threshold test to decide whether to accept a swap.
    """
    delta_target = 1 / (objective.num_users ** 1.5)
    T = calculate_iterations(k, gamma)
    # Budget split: 50% for initial/final steps, 50% for iterative steps
    comp = 2 * T
    eps_0 = get_best_eps_0(eps_target=eps / 3, delta_target=delta_target, k=comp, decomposable=False)
    forced_exploration = min(10, T)

    S = get_initial_set(objective, ground_set, partition_map, partition_limits, k, eps=eps / 3, private=True)
    current_val, coverage, dist, _ = objective.evaluate(S, distort=False)
    observed_sets = {tuple(sorted(S)): current_val}

    sample_size = math.ceil(ground_set.nb_elements / k)
    lap_scale = objective.sensitivity * (2 + gamma / ground_set.nb_elements) / eps_0 if eps_0 > 0 else 0
    partition_counts = {p: 0 for p in partition_limits}
    for e in S:
        partition_counts[partition_map[e]] += 1
    for it in range(int(T)):
        S_set = set(S)
        sampled_elements = random.sample(ground_set.elements, min(len(ground_set.elements), sample_size))
        feasible_swaps = []
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in sampled_elements:
                if e_in in S_set: continue
                p_in_id = partition_map[e_in]
                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) < partition_limits[p_in_id]:
                    feasible_swaps.append((e_out, e_in))

        if not feasible_swaps:
            break

        sampled_swap_values = {}
        for e_out, e_in in feasible_swaps:
            idx = S.index(e_out)
            S_tmp = S[:idx] + S[idx + 1:] + [e_in]
            sampled_swap_values[(e_out, e_in)] = objective.evaluate(S_tmp, distort=False)[0]

        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        new_val = sampled_swap_values[best_swap]

        laplace_noise = np.random.laplace(0, lap_scale) if private else 0
        if new_val - (1 + gamma / ground_set.nb_elements) * current_val > laplace_noise or it < forced_exploration:
            e_out, e_in = best_swap
            idx = S.index(e_out)
            S = S[:idx] + S[idx + 1:] + [e_in]
            partition_counts[partition_map[e_out]] -= 1
            partition_counts[partition_map[e_in]] += 1
            current_val = new_val
            observed_sets[tuple(sorted(S))] = current_val
            print('current_val:', current_val)
        else:
            break

    S_best = list(exp_mech(observed_sets, eps/3, objective.sensitivity, private=private))
    val, cov, div, _ = objective.evaluate(S_best, distort=False)
    print(f'val: {val}')
    return S_best, val, cov, div, objective.num_queries


def random_baseline(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k):
    """
    Generates a random feasible set that satisfies the partition matroid constraints.
    """
    elements = list(ground_set.elements)
    random.shuffle(elements)

    S = []
    counts = {p: 0 for p in partition_limits}

    for e in elements:
        if len(S) >= k:
            break

        p_id = partition_map.get(e)
        # Check if the store (partition) has remaining capacity
        if p_id in partition_limits and counts[p_id] < partition_limits[p_id]:
            S.append(e)
            counts[p_id] += 1

    # Evaluation
    if len(S) < k:
        print(f"Baseline Warning: Could only find {len(S)} feasible elements.")

    val, rel, div, _ = objective.evaluate(S, distort=False)

    return S, val, rel, div, objective.num_queries
