from classes import MSDObjective, GroundSet


def greedy(objective: MSDObjective, ground_set: GroundSet, k):
    """
    Standard Greedy Algorithm for  maximization (Borodin et al.).

    Args:
        objective: An instance of MSDFacilityLocation
        ground_set: An instance of GroundSet
        k: Cardinality constraint

    Returns:
        S: The list of selected element indices
        current_val: The final objective value
    """
    objective.distortion = 1
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
            gain, potential_aux = objective.marginal_gain(e, S, current_val, auxiliary)

            if gain > best_gain:
                best_gain = gain
                best_e = e
                best_aux = potential_aux

        # If no improvement is possible, stop early
        if best_e is None or best_gain <= 0:
            break

        # Commit the best choice: update value and the 'snapshot' (auxiliary)
        # Note: We use the already calculated best_aux to avoid re-calculating
        current_val += best_gain
        auxiliary = best_aux

        S.append(best_e)
        remaining_elements.remove(best_e)

        print(f"Iteration {i + 1}: Added {best_e}, Total Value: {current_val:.4f}")

    return S, current_val
