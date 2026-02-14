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


def get_arbitrary_base(ground_set, partition_map, partition_limits, k):
    """
    Finds an arbitrary base (size k) for the matroid intersection.
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


def local_search(objective, ground_set: GroundSet, partition_map, partition_limits, k, gamma):

    objective.num_queries = 0
    S = get_initial_set(objective, ground_set, partition_map, partition_limits, k)
    print('Got initial set', S)

    # 1. Capture the initial auxiliary state
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)

    partition_counts = {p: 0 for p in partition_limits}
    for e in S:
        partition_counts[partition_map[e]] += 1

    while True:
        swap_values = {}
        S_set = set(S)

        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in ground_set.elements:
                if e_in in S_set:
                    continue
                p_in_id = partition_map[e_in]

                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) + 1 <= partition_limits[p_in_id]:
                    # 2. evaluate_swap uses the persistent auxiliary state
                    swap_values[(e_out, e_in)] = objective.evaluate_swap(e_out, e_in, S, auxiliary, distort=False)

        if not swap_values:
            break

        best_swap = max(swap_values, key=swap_values.get)
        best_val = swap_values[best_swap]

        if best_val > current_val * (1 + gamma / k):
            e_out, e_in = best_swap

            # 3. COMMIT THE SWAP via the Objective Class
            # This updates the counts and distance sum without re-evaluating everything
            auxiliary = objective.swap_element(e_out, e_in, S, auxiliary)

            idx = S.index(e_out)
            S = S[:idx] + S[idx + 1:] + [e_in]

            partition_counts[partition_map[e_out]] -= 1
            partition_counts[partition_map[e_in]] += 1

            current_val = best_val
            print(f"Committed: {e_out} -> {e_in} | New Val: {current_val:.4f}")
        else:
            break

    # Re-eval one last time for final return stats
    current_val, coverage, dist, _ = objective.evaluate(S, distort=False)
    print(f"Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}, Selected: {S}")
    return S, current_val, coverage, dist, objective.num_queries


def DP_sample_local_search(objective, ground_set, partition_map, partition_limits, k, eps, gamma, private):
    """
    DP Local Search using sampling to reduce sensitivity and query count.
    Updates the set at every iteration for T iterations.
    """
    objective.num_queries = 0
    delta_target = 1 / (objective.num_users ** 1.5)
    T = calculate_iterations(k, gamma)
    eps_0 = get_best_eps_0(eps_target=eps/2, delta_target=delta_target, k=T, decomposable=False)

    S = get_arbitrary_base(ground_set, partition_map, partition_limits, k)
    print('Got initial set', S)
    # 1. Capture the initial auxiliary state
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)

    observed_sets = {tuple(sorted(S)): current_val}
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
            sampled_swap_values[(e_out, e_in)] = objective.evaluate_swap(e_out, e_in, S, auxiliary, distort=False)

        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        e_out, e_in = best_swap
        auxiliary = objective.swap_element(e_out, e_in, S, auxiliary)
        idx = S.index(e_out)
        S = S[:idx] + S[idx + 1:] + [e_in]
        partition_counts[partition_map[e_out]] -= 1
        partition_counts[partition_map[e_in]] += 1

        current_val = sampled_swap_values[best_swap]
        observed_sets[tuple(sorted(S))] = current_val
        # print('current_val:', current_val)

    S_best = list(exp_mech(observed_sets, eps/2, objective.sensitivity, private=private))
    val, cov, div, _ = objective.evaluate(S_best, distort=False)
    print(f"Total Value: {val:.4f}, Coverage: {cov:.4f}, Diversity: {div:.4f}, Selected: {S}")
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

    val, cov, div, _ = objective.evaluate(S, distort=False)
    print(f"Total Value: {val:.4f}, Coverage: {cov:.4f}, Diversity: {div:.4f}")
    return S, val, cov, div, objective.num_queries
