import random

import numpy as np

from classes import MSDObjective, GroundSet
from dp_mechanisms import exp_mech

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
    S = []
    # Initialize state: f(empty_set) = 0
    current_val, auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):
        best_gain = -float('inf')
        best_e = None
        best_aux = None

        # Scan all available candidates
        for e in remaining_elements:
            # Calculate gain using our memory-efficient logic
            gain, potential_aux = objective.marginal_gain(e, S, auxiliary)

            if gain >= best_gain:
                best_gain = gain
                best_e = e
                best_aux = potential_aux

        # # If no improvement is possible, stop early
        # if best_e is None or best_gain <= 0:
        #     break

        # Commit the best choice: update value and the 'snapshot' (auxiliary)
        # Note: We use the already calculated best_aux to avoid re-calculating
        current_val += best_gain
        auxiliary = best_aux

        S.append(best_e)
        remaining_elements.remove(best_e)

        print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val, _ = objective.evaluate(S)
    return S, val


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
    S = []
    # Initialize state: f(empty_set) = 0
    current_val, auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):

        candidates_scores = {
            e: objective.marginal_gain(e, S, auxiliary)[0]
            for e in remaining_elements
        }

        best_e = exp_mech(candidates_scores, eps, objective.sensitivity, private=private)
        best_gain, best_aux = objective.marginal_gain(best_e, S, auxiliary)
        current_val += best_gain
        auxiliary = best_aux
        S.append(best_e)
        remaining_elements.remove(best_e)

        print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val,_ = objective.evaluate(S)
    return S, val

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
    objective.distortion = 1 if oblivious else 1/(2+gamma)
    S = []
    # Initialize state: f(empty_set) = 0
    current_val, auxiliary = objective.evaluate(S)

    # Set of candidates still available to pick
    remaining_elements = set(ground_set.elements)

    for i in range(k):

        # subsample
        g_i = min(k, ground_set.nb_elements-i+1) if not oblivious else k-i+1
        sample_size = int(np.ceil(len(remaining_elements) * min( np.log(1/gamma)/g_i ,1.0)))
        sampled_elements = set(random.sample(remaining_elements, sample_size))

        candidates_scores = {
            e: objective.marginal_gain(e, S, auxiliary)[0]
            for e in sampled_elements
        }

        best_e = exp_mech(candidates_scores, eps, objective.sensitivity, private=private)
        best_gain, best_aux = objective.marginal_gain(best_e, S, auxiliary)
        current_val += best_gain
        auxiliary = best_aux
        S.append(best_e)
        remaining_elements.remove(best_e)

        print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    objective.distortion = 1
    val,_ = objective.evaluate(S)
    return S, val
