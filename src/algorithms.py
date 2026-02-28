import random
import numpy as np
import math
from dp_mechanisms import exp_mech, get_best_eps_0
from classes import MSDObjective, GroundSet, MSDAmazonObjective


def greedy(objective: MSDObjective, ground_set: GroundSet, k):
    """
    Non-private Distorted Greedy Algorithm (Borodin et al.).

    Args:
        objective: An instance of MSDFacilityLocation
        ground_set: An instance of GroundSet
        k: Cardinality constraint

    Returns:
        S: The list of selected element indices
        current_val: The final objective value
    """
    objective.distortion = 0.5
    objective.num_queries = 0
    S = []
    # Initialize state: f(empty_set) = 0
    current_val, _, __, auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):
        best_gain = -float('inf')
        best_e = None

        # Scan all available candidates
        for e in remaining_elements:
            # Calculate gain using our memory-efficient logic
            gain, _ = objective.marginal_gain(e, S, auxiliary)

            if gain >= best_gain:
                best_gain = gain
                best_e = e

        # ONLY NOW do we copy the array and update the state (Once per K, not per E)
        auxiliary = objective.add_one_element(best_e, S, auxiliary)

        # # If no improvement is possible, stop early
        # if best_e is None or best_gain <= 0:
        #     break

        # Commit the best choice: update value and the 'snapshot' (auxiliary)
        # Note: We use the already calculated best_aux to avoid re-calculating
        current_val += best_gain
        S.append(best_e)
        remaining_elements.remove(best_e)

        # print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val, coverage, dist, _ = objective.evaluate(S, distort=False)
    print(f"Total Value: {val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
    return S, val, coverage, dist, objective.num_queries


def DP_greedy(objective: MSDObjective, ground_set: GroundSet, k, eps, private):
    """
    Private Max-Sum Greedy (Algorithm 1)

    Args:
        objective: An instance of MSDFacilityLocation
        ground_set: An instance of GroundSet
        k: Cardinality constraint
        eps: Privacy parameter

    Returns:
        S: The list of selected element indices
        current_val: The final objective value
    """
    objective.distortion = 0.5
    objective.num_queries = 0
    S = []
    # Initialize state: f(empty_set) = 0
    current_val, _, __, auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):

        candidates_scores = {
            e: objective.marginal_gain(e, S, auxiliary)[0]
            for e in remaining_elements
        }
        best_e = exp_mech(candidates_scores, eps, objective.sensitivity, private=private)
        best_gain, _ = objective.marginal_gain(best_e, S, auxiliary, charge=False)
        auxiliary = objective.add_one_element(best_e, S, auxiliary)
        current_val += best_gain
        S.append(best_e)
        remaining_elements.remove(best_e)

        # print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val, coverage, dist, _ = objective.evaluate(S, distort=False)

    print(f"Total Value: {val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
    return S, val, coverage, dist, objective.num_queries

def DP_sample_greedy(objective: MSDObjective, ground_set: GroundSet, k, eps, private, oblivious, gamma):
    """
    Private Max-Sum Greedy (Algorithm 1)

    Args:
        objective: An instance of MSDFacilityLocation
        ground_set: An instance of GroundSet
        k: Cardinality constraint
        eps: Privacy parameter
        private: Whether the algorithm is executed with DP noise
        oblivious:
        gamma: Utility parameter

    Returns:
        S: The list of selected element indices
        current_val: The final objective value
    """
    objective.distortion = 1 if oblivious else 1/(2-gamma)
    objective.num_queries = 0

    S = []
    # Initialize state: f(empty_set) = 0
    current_val, _, __,  auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):

        # subsample
        g_i = min(k, ground_set.nb_elements-i+1) if oblivious else k-i+1
        sample_size = int(np.ceil(len(remaining_elements) * min( np.log(1/gamma)/g_i, 1.0)))
        sampled_elements = set(random.sample(remaining_elements, sample_size))

        candidates_scores = {
            e: objective.marginal_gain(e, S, auxiliary)[0]
            for e in sampled_elements
        }

        best_e = exp_mech(candidates_scores, eps, objective.sensitivity, private=private)
        best_gain, _ = objective.marginal_gain(best_e, S, auxiliary)
        current_val += best_gain
        auxiliary = objective.add_one_element(best_e, S, auxiliary)
        S.append(best_e)
        remaining_elements.remove(best_e)

        # print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val, coverage, dist, _ = objective.evaluate(S, distort=False)

    print(f"Total Value: {val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}")
    return S, val, coverage, dist, objective.num_queries


def random_baseline(objective: MSDObjective, ground_set: GroundSet, k):
    """
    Random Baseline Algorithm.
    Selects a random feasible subset of size k and evaluates it.

    Args:
        objective: An instance of MSDFacilityLocation
        ground_set: An instance of GroundSet
        k: Cardinality constraint

    Returns:
        S: The list of randomly selected element indices
        final_val: The objective value of the random set
    """
    objective.distortion = 1.0

    S = random.sample(list(ground_set.elements), k)

    val, rel, div, _ = objective.evaluate(S, distort=False)
    print(f"Total Value: {val:.4f}, Coverage: {rel:.4f}, Diversity: {div:.4f}")
    return S, val, rel, div


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


def get_initial_set_top1(objective, ground_set, partition_map, partition_limits, k, eps=None, private=False):
    # 1. Evaluate all singletons that fit in a partition
    scores = {e: objective.evaluate([e], distort=False)[0]
              for e in ground_set.elements if partition_limits.get(partition_map[e], 0) >= 1}

    # 2. Select top element (Private or Exact)
    best_e = exp_mech(scores, eps, objective.sensitivity, private=private) if private else max(scores, key=scores.get)

    # 3. Matroid completion (fill remaining k-1 slots)
    S, counts = [best_e], {p: 0 for p in partition_limits}
    counts[partition_map[best_e]] = 1

    for e in ground_set.elements:
        p = partition_map[e]
        if len(S) < k and e not in S and counts.get(p, 0) < partition_limits.get(p, 0):
            S.append(e)
            counts[p] = counts.get(p, 0) + 1

    return S

def get_initial_top_two_base(objective :MSDAmazonObjective, ground_set, partition_map, partition_limits, k, eps=None, private=False):
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
    S = get_initial_top_two_base(objective, ground_set, partition_map, partition_limits, k)
    # print('Got initial set', S)

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
    # print(f"Total Value: {current_val:.4f}, Coverage: {coverage:.4f}, Diversity: {dist:.4f}, Selected: {S}")
    return S, current_val, coverage, dist, objective.num_queries

def DP_sample_local_search(objective: MSDAmazonObjective, ground_set, partition_map, partition_limits, k, eps, gamma, private):
    """
    DP Local Search using sampling to reduce sensitivity and query count.
    Updates the set at every iteration for T iterations.
    """
    objective.num_queries = 0
    delta_target = 1 / (objective.num_users ** 1.5)
    T = calculate_iterations(k, gamma)
    # print(f'iterations: {T}')
    eps_0 = get_best_eps_0(eps_target=eps/2, delta_target=delta_target, k=T+1, decomposable=False)

    # S = get_arbitrary_base(ground_set, partition_map, partition_limits, k)
    S = get_initial_set_top1(objective, ground_set, partition_map, partition_limits, k, eps_0, private=private)
    # S = get(objective, ground_set, partition_map, partition_limits, k, eps / 3, private=private)
    # print('Got initial set', S)
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
        sampled_ground_set = random.sample(ground_set.elements, sample_size)
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
            sampled_swap_values[(e_out, e_in)] = objective.evaluate_swap(e_out, e_in, S, auxiliary, distort=False)

        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        e_out, e_in = best_swap
        auxiliary = objective.swap_element(e_out, e_in, S, auxiliary)
        idx = S.index(e_out)
        S = S[:idx] + S[idx + 1:] + [e_in]
        partition_counts[partition_map[e_out]] -= 1
        partition_counts[partition_map[e_in]] += 1

        current_val = sampled_swap_values[best_swap]
        # if it % 50 == 0:
        #     current_val, _, _, auxiliary = objective.evaluate(S, distort=False)

        observed_sets[tuple(sorted(S))] = current_val
        # print('current_val:', current_val)

    S_best = list(exp_mech(observed_sets, eps/2, objective.sensitivity, private=private))
    val, cov, div, _ = objective.evaluate(S_best, distort=False)
    # print(f"Total Value: {val:.4f}, Coverage: {cov:.4f}, Diversity: {div:.4f}, Selected: {S}")
    return S_best, val, cov, div, objective.num_queries

def DP_local_search(objective: MSDAmazonObjective, ground_set, partition_map, partition_limits, k, eps, gamma, private):

    objective.num_queries = 0
    delta_target = 1 / (objective.num_users ** 1.5)
    T = calculate_iterations(k, gamma)
    # print(f'iterations: {T}')
    eps_0 = get_best_eps_0(eps_target=eps/2, delta_target=delta_target, k=T+1, decomposable=False)

    # S = get_arbitrary_base(ground_set, partition_map, partition_limits, k)
    S = get_initial_set_top1(objective, ground_set, partition_map, partition_limits, k, eps_0, private=private)
    # S = get(objective, ground_set, partition_map, partition_limits, k, eps / 3, private=private)
    # print('Got initial set', S)
    # 1. Capture the initial auxiliary state
    current_val, coverage, dist, auxiliary = objective.evaluate(S, distort=False)

    observed_sets = {tuple(sorted(S)): current_val}

    partition_counts = {p: 0 for p in partition_limits}
    for e in S:
        partition_counts[partition_map[e]] += 1

    for it in range(int(T)):
        S_set = set(S)
        feasible_swaps = []
        sampled_ground_set = ground_set.elements
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
            sampled_swap_values[(e_out, e_in)] = objective.evaluate_swap(e_out, e_in, S, auxiliary, distort=False)

        best_swap = exp_mech(sampled_swap_values, eps_0, objective.sensitivity, private=private)
        e_out, e_in = best_swap
        auxiliary = objective.swap_element(e_out, e_in, S, auxiliary)
        idx = S.index(e_out)
        S = S[:idx] + S[idx + 1:] + [e_in]
        partition_counts[partition_map[e_out]] -= 1
        partition_counts[partition_map[e_in]] += 1

        current_val = sampled_swap_values[best_swap]
        # if it % 50 == 0:
        #     current_val, _, _, auxiliary = objective.evaluate(S, distort=False)

        observed_sets[tuple(sorted(S))] = current_val
        # print('current_val:', current_val)

    S_best = list(exp_mech(observed_sets, eps/2, objective.sensitivity, private=private))
    val, cov, div, _ = objective.evaluate(S_best, distort=False)
    # print(f"Total Value: {val:.4f}, Coverage: {cov:.4f}, Diversity: {div:.4f}, Selected: {S}")
    return S_best, val, cov, div, objective.num_queries

def random_baseline_matroid(objective: MSDObjective, ground_set: GroundSet, partition_map, partition_limits, k):
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

    # # Evaluation
    # if len(S) < k:
    #     print(f"Baseline Warning: Could only find {len(S)} feasible elements.")

    val, cov, div, _ = objective.evaluate(S, distort=False)
    # print(f"Total Value: {val:.4f}, Coverage: {cov:.4f}, Diversity: {div:.4f}")
    return S, val, cov, div, 0


