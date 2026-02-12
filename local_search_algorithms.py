import math
import random

import numpy as np

from classes import MSDObjective, GroundSet
from dp_mechanisms import exp_mech, get_best_eps_0


def calculate_iterations(k, gamma):
    """
    Calculates the iteration threshold T.
    Formula: T = ceil((2 * r * log(r)) / (gamma * (1 - 1/e)))
    """
    # math.log defaults to natural log (base e)
    numerator = 2 * k * math.log(k)
    denominator = gamma * (1 - 1 / math.e)

    T = math.ceil(numerator / denominator)
    return T


def get_initial_set(objective, ground_set, partition_map, partition_limits, k, eps=None, private=False):
    elements = ground_set.elements
    feasible_pair_values = {}

    # 1. Collect values for all feasible pairs
    for i, e1 in enumerate(elements):
        p1 = partition_map[e1]
        if partition_limits[p1] < 1:
            continue

        for e2 in elements[i + 1:]:
            p2 = partition_map[e2]
            # Check feasibility: if same partition, need limit >= 2; else both need >= 1
            if (p1 == p2 and partition_limits[p1] >= 2) or (p1 != p2 and partition_limits[p2] >= 1):
                val = objective.evaluate([e1, e2], distort=False)[0]
                feasible_pair_values[(e1, e2)] = val

    # 2. Select the pair (Deterministically or Privately)
    if private:
        if eps is None:
            raise ValueError("eps_0 must be provided for private initialization.")
        # Use Exponential Mechanism to pick the pair
        best_pair_tuple = exp_mech(feasible_pair_values, eps, objective.sensitivity, private=private)
        best_pair = list(best_pair_tuple)
    else:
        # Standard max selection
        best_pair_tuple = max(feasible_pair_values, key=feasible_pair_values.get)
        best_pair = list(best_pair_tuple)

    # 3. Complete to base arbitrarily (This part doesn't look at data, so it's "free" privacy-wise)
    print(best_pair)
    S = list(best_pair)
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
# def get_initial_set(objective, ground_set, partition_map, partition_limits, k, private = False):
#     elements = ground_set.elements
#     best_pair, max_val = [], -float('inf')
#
#     # 1. Find the best feasible pair
#     for i, e1 in enumerate(elements):
#         p1 = partition_map[e1]
#         if partition_limits[p1] < 1: continue
#
#         for e2 in elements[i + 1:]:
#             p2 = partition_map[e2]
#             # Check feasibility: if same partition, need limit >= 2; else both need >= 1
#             if (p1 == p2 and partition_limits[p1] >= 2) or (p1 != p2 and partition_limits[p2] >= 1):
#                 val = objective.evaluate([e1, e2], distort=False)[0]
#                 if val > max_val:
#                     max_val, best_pair = val, [e1, e2]
#
#     # 2. Complete to base arbitrarily
#     S = list(best_pair)
#     counts = {p: 0 for p in partition_limits}
#     for e in S: counts[partition_map[e]] += 1
#
#     for e in elements:
#         if len(S) >= k: break
#         p = partition_map[e]
#         if e not in S and counts[p] < partition_limits[p]:
#             S.append(e)
#             counts[p] += 1
#
#     return S


def get_arbitrary_feasible(ground_set, partition_map, partition_limits, k):
    """
    Finds an arbitrary feasible base (size k) for the matroid intersection.
    """
    S = []
    # Initialize counts for each partition
    counts = {p: 0 for p in partition_limits}

    for e in ground_set.elements:
        if len(S) == k:
            break

        p_id = partition_map.get(e)

        # Check if the element's partition has remaining capacity
        # and we haven't reached the global cardinality k
        if p_id in partition_limits and counts[p_id] < partition_limits[p_id]:
            S.append(e)
            counts[p_id] += 1

    if len(S) < k:
        print(f"Warning: Could only find {len(S)} feasible elements out of {k} requested.")

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


def DP_sample_local_search(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k, eps,
                           gamma, private):

    # Compute number of compositions
    delta_target = 1 / (objective.num_users ** 1.5)
    comp = calculate_iterations(k, gamma) + 1
    eps_0 = get_best_eps_0(eps_target=eps, delta_target=delta_target, k=comp,  decomposable=False)

    # Step 0: Custom Initialization
    initial_S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
    S = list(initial_S)
    observed_sets = dict()

    # Initial evaluation
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
    observed_sets[tuple(S)] = current_val

    T = calculate_iterations(k, gamma)
    # sample_size determines how many heavy evaluations we do per iteration
    sample_size = math.ceil(ground_set.nb_elements / k)

    print(f'Will perform {T} iterations. Sample size per iteration: {sample_size}')

    for it in range(int(T)):
        S_set = set(S)
        partition_counts = {p: 0 for p in partition_limits}
        for e in S:
            partition_counts[partition_map[e]] += 1

        # 1. Identify ALL feasible swap pairs (FAST: no objective calls here)
        feasible_swaps = []
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in ground_set.elements:
                if e_in in S_set:
                    continue

                p_in_id = partition_map[e_in]
                if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) < partition_limits[p_in_id]:
                    feasible_swaps.append((e_out, e_in))

        if not feasible_swaps:
            break

        # 2. Subsample from feasible swaps BEFORE evaluating (CRITICAL FOR SPEED)
        actual_size = min(len(feasible_swaps), sample_size)
        sampled_keys = random.sample(feasible_swaps, actual_size)

        # 3. Evaluate ONLY the sampled swaps (HEAVY: limited to sample_size)
        sampled_swap_values = {}
        for e_out, e_in in sampled_keys:
            idx = S.index(e_out)
            S_tmp = S[:idx] + S[idx + 1:] + [e_in]
            val, _, _, _ = objective.evaluate(S_tmp, distort=False)
            sampled_swap_values[(e_out, e_in)] = val

        # 4. Select swap privately
        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)

        # 5. Commit swap
        e_out, e_in = best_swap
        idx = S.index(e_out)
        S = S[:idx] + S[idx + 1:] + [e_in]

        # Re-computation for the next step trajectory
        current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
        observed_sets[tuple(S)] = current_val

        if it % 10 == 0:
            print(f"Iteration {it}/{int(T)} - Current Val: {current_val:.4f}")

    # Final Step: Pick the best set found across all iterations privately
    S_best_tuple = exp_mech(observed_sets, eps_0, objective.sensitivity, private=private)
    S_best = list(S_best_tuple)

    current_val, coverage, dist, _ = objective.evaluate(S_best, distort=False)
    print(f"Final Selection - Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")

    return S_best, current_val, coverage, dist, objective.num_queries


def DP_sample_local_search_threshold(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k,
                                     eps, gamma, private):
    # Compute number of compositions
    delta_target = 1 / (objective.num_users ** 1.5)
    comp = 2 * calculate_iterations(k, gamma) + 1  # T iterations each involves EM and Laplace. Final EM is + 1.
    eps_0 = get_best_eps_0(eps_target=eps/2, delta_target=delta_target, k=comp, decomposable=False)

    # Step 0: Custom Initialization
    # initial_S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
    initial_S = get_initial_set(objective, ground_set, partition_map, partition_limits, k, eps=eps/2,private=True)
    S = list(initial_S)
    observed_sets = dict()

    # Initial evaluation
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
    observed_sets[tuple(S)] = current_val

    T = calculate_iterations(k, gamma)
    sample_size = math.ceil(ground_set.nb_elements / k)

    # Sensitivity/eps for Laplace noise
    lap_scale = objective.sensitivity*(2+gamma/k) / eps_0 if eps_0 > 0 else 0

    print(f'Will perform {T} iterations. Sample size: {sample_size}')

    for it in range(int(T)):
        S_set = set(S)
        partition_counts = {p: 0 for p in partition_limits}
        for e in S:
            partition_counts[partition_map[e]] += 1

        sampled_ground_set = random.sample(ground_set.elements, sample_size)

        feasible_swaps = []
        for e_out in S:
            p_out_id = partition_map[e_out]
            for e_in in sampled_ground_set:
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
            val, _, _, _ = objective.evaluate(S_tmp, distort=False)
            sampled_swap_values[(e_out, e_in)] = val

        # 1. Select potential swap privately
        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        new_val = sampled_swap_values[best_swap]

        # 2. Threshold Test with Laplace Noise
        # Difference = new_val - current_val
        # We test if: Difference + Laplace(sensitivity/eps) > 0
        laplace_noise = np.random.laplace(0, lap_scale) if private else 0
        # laplace_noise=0
        if new_val - (1+gamma/k)*current_val > laplace_noise:
            # 3. Commit swap
            e_out, e_in = best_swap
            idx = S.index(e_out)
            S = S[:idx] + S[idx + 1:] + [e_in]
            current_val = new_val  # Updated to the value we already calculated
            observed_sets[tuple(S)] = current_val
            if it % 10 == 0:
                print(f"Iteration {it}: Swap accepted. New Val: {current_val:.4f}")
        else:
            print(f"{new_val - (1+gamma/k)*current_val }, Lap: {laplace_noise} ")
            break

            # Final Step: Pick the best set found across all iterations privately
    S_best_tuple = exp_mech(observed_sets, eps_0, objective.sensitivity, private=private)
    S_best = list(S_best_tuple)

    current_val, coverage, dist, _ = objective.evaluate(S_best, distort=False)
    print(f"Final Selection - Total Value: {current_val:.4f}")

    return S_best, current_val, coverage, dist, objective.num_queries


# def DP_sample_local_search(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k, eps,
#                            gamma, private):
#     # Step 0: Custom Initialization
#     initial_S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
#     S = list(initial_S)
#     observed_sets = dict()
#
#     # Initial evaluation
#     current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
#     # Lists aren't hashable; must use tuple for dict keys
#     observed_sets[tuple(S)] = current_val
#
#     T = calculate_iterations(k, gamma)
#     sample_size = math.ceil(ground_set.nb_elements / k)
#     print(f'Will perform {T} iterations')
#     for it in range(int(T)):
#         print(it)
#         swap_values = {}
#         S_set = set(S)
#
#         partition_counts = {p: 0 for p in partition_limits}
#         for e in S:
#             partition_counts[partition_map[e]] += 1
#
#         # 1. Build dictionary of feasible swaps
#         for e_out in S:
#             p_out_id = partition_map[e_out]
#             for e_in in ground_set.elements:
#                 if e_in in S_set: continue
#                 p_in_id = partition_map[e_in]
#
#                 # Feasibility check
#                 if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) < partition_limits[p_in_id]:
#                     idx = S.index(e_out)
#                     S_tmp = S[:idx] + S[idx + 1:] + [e_in]
#                     val, _, _, _ = objective.evaluate(S_tmp, distort=False)
#                     swap_values[(e_out, e_in)] = val
#
#         if not swap_values:
#             break
#
#         actual_size = min(len(swap_values), sample_size)
#         sampled_keys = random.sample(list(swap_values.keys()), actual_size)
#         # Using 'swp' instead of 'k' to avoid shadowing the 'k' parameter
#         sampled_swap_values = {swp: swap_values[swp] for swp in sampled_keys}
#
#         # 2. Select swap privately
#         best_swap = exp_mech(sampled_swap_values, eps, objective.sensitivity, private=private)
#
#         # 3. Commit swap
#         e_out, e_in = best_swap
#         idx = S.index(e_out)
#         S = S[:idx] + S[idx + 1:] + [e_in]
#
#         # Re-computation for the next step
#         current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
#         observed_sets[tuple(S)] = current_val
#
#     # Final Step: Pick the best set found across all iterations privately
#     S_best_tuple = exp_mech(observed_sets, eps, objective.sensitivity, private=private)
#     S_best = list(S_best_tuple)
#
#     current_val, coverage, dist, _ = objective.evaluate(S_best, distort=False)
#     print(f"Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
#
#     return S_best, current_val, coverage, dist, objective.num_queries

# def DP_sample_local_search(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k, eps, gamma, private):
#
#     # Step 0: Custom Initialization
#     initial_S = get_arbitrary_feasible(ground_set, partition_map, partition_limits, k)
#     S = list(initial_S)
#     observed_sets = dict()
#
#     # Initial evaluation
#     current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)
#     observed_sets[S] = current_val
#     T = calculate_iterations(k, gamma)
#     sample_size = math.ceil(ground_set.nb_elements / k)
#
#     for it in range(T):
#
#         swap_values = {}
#         S_set = set(S)
#
#         partition_counts = {p: 0 for p in partition_limits}
#         for e in S:
#             p_id = partition_map[e]
#             partition_counts[p_id] += 1
#
#         # 1. Build dictionary of feasible swaps
#         for e_out in S:
#             p_out_id = partition_map[e_out]
#             for e_in in ground_set.elements:
#                 if e_in in S_set:
#                     continue
#
#                 p_in_id = partition_map[e_in]
#
#                 # Feasibility check
#                 if p_in_id == p_out_id or partition_counts.get(p_in_id, 0) + 1 <= partition_limits[p_in_id]:
#                     idx = S.index(e_out)
#                     S_tmp = S[:idx] + S[idx + 1:] + [e_in]
#                     val, _, _, _ = objective.evaluate(S_tmp, distort=False)
#                     swap_values[(e_out, e_in)] = val
#
#         if not swap_values:
#             break
#         # Subsample
#         sampled_keys = random.sample(list(swap_values.keys()), sample_size)
#         sampled_swap_values = {k: swap_values[k] for k in sampled_keys}
#
#         # 2. Select best swap
#         best_swap = exp_mech(sampled_swap_values, eps, objective.sensitivity, private=private)
#
#         # 3. Test threshold and commit
#         e_out, e_in = best_swap
#         idx = S.index(e_out)
#         S_tmp = S[:idx] + S[idx + 1:] + [e_in]
#
#         # Final state re-computation
#         current_val, coverage, dist, auxiliary = objective.evaluate(S_tmp, distort=False)
#         S = S_tmp
#         observed_sets[S] = current_val
#
#     S_best = exp_mech(observed_sets, eps, objective.sensitivity, private=private)
#     current_val, coverage, dist, auxiliary = objective.evaluate(S_best, distort=False)
#
#     print(f"Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
#     return S_best, current_val, coverage, dist, objective.num_queries
#
#

def random_baseline(objective: MSDObjective, ground_set: GroundSet, k):
    pass