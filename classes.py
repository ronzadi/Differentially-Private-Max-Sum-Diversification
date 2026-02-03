from abc import ABC, abstractmethod
import numpy as np

class GroundSet:
    def __init__(self, elements):
        """
        Args:
            elements: A list of indices or objects (e.g., [0, 1, 2, ..., 19])
        """
        self.elements = elements
        self.nb_elements = len(elements)

    def __iter__(self):
        return iter(self.elements)


class MSDObjective(ABC):
    """
    Abstract Base Class for submodular+MSD functions.
    """

    @abstractmethod
    def evaluate(self, S):
        """
        Calculates f(S).
        Returns: (value, auxiliary)
        """
        pass

    @abstractmethod
    def marginal_gain(self, e, S, auxiliary):
        """
        Calculates f(S U {e}) - f(S).
        Returns: (gain, new_auxiliary)
        """
        pass

    def add_one_element(self, e, S, auxiliary):
        """
        A default implementation for adding an element.
       use this to update state after a selection.
        """
        gain, new_auxiliary = self.marginal_gain(e, S, auxiliary)
        return gain, new_auxiliary


import numpy as np


class MSDFacilityLocation(MSDObjective):  # Inherits from MSDObjective if required by your script
    def __init__(self, passenger_coords, grid_coords, lambda_param, k, distortion, sensitivity):
        """
        Args:
            passenger_coords: [N x 2] array of Uber pickup locations
            grid_coords: [M x 2] array of candidate hub locations
            lambda_param: Weighting between coverage and diversity
            k: Cardinality constraint (total hubs to select)
            distortion: Multiplier for the submodular term (usually 1/2 in greedy)
        """
        self.passengers = passenger_coords
        self.grid = grid_coords
        self.lambda_param = lambda_param
        self.k = k
        self.distortion = distortion
        self.sensitivity = sensitivity

        # Hardcoded normalization constant
        self.m_constant = 0.2

        self.num_passengers = len(passenger_coords)
        # Denominator for diversity (Max-Sum)
        self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1

    def _get_similarity_row(self, grid_idx):
        """
        Calculates 1 - M(i, j) where M(i, j) is normalized Manhattan distance.
        Matches the utility function logic: fD(S) = sum(1 - min M(l, p)).
        """
        # L1 Distance: |i1 - j1| + |i2 - j2|
        diffs = np.abs(self.passengers - self.grid[grid_idx])
        l1_dists = np.sum(diffs, axis=1)

        # Normalize by m
        normalized_dists = l1_dists / self.m_constant

        # Return 1 - normalized_dist (capped at 0)
        return np.maximum(0, 1.0 - normalized_dists)

    def evaluate(self, S):
        """
        Full evaluation of a set S. Required by the Greedy algorithm to initialize.
        """
        if not S:
            # Auxiliary state: (max_similarities_per_passenger, sum_of_distances_between_hubs)
            return 0.0, (np.zeros(self.num_passengers), 0.0)

        # 1. Coverage (Facility Location) term
        max_sims = np.zeros(self.num_passengers)
        for idx in S:
            max_sims = np.maximum(max_sims, self._get_similarity_row(idx))

        coverage_term = np.sum(max_sims) / self.num_passengers

        # 2. Diversity (Max-Sum) term
        dist_sum = 0.0
        n_S = len(S)
        if n_S > 1:
            for i in range(n_S):
                for j in range(i + 1, n_S):
                    diff = np.abs(self.grid[S[i]] - self.grid[S[j]])
                    dist_sum += np.sum(diff) / self.m_constant

        avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        # Weighted combination
        total_val = (1 - self.lambda_param) * self.distortion * coverage_term + (self.lambda_param * avg_dist)

        return total_val, (max_sims, dist_sum)

    def marginal_gain(self, e, S, auxiliary):
        """
        Efficiently calculates the increase in utility if element 'e' is added to 'S'.
        """
        max_sims, current_dist_sum = auxiliary

        # 1. Coverage Gain
        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)
        # Benefit is the sum of improvements across all passengers
        coverage_gain = (np.sum(new_max_sims) - np.sum(max_sims)) / self.num_passengers

        # 2. Diversity Gain
        dist_to_existing = 0.0
        if S:
            # Manhattan distance from new candidate 'e' to all selected 'S'
            hub_diffs = np.abs(self.grid[S] - self.grid[e])
            dist_to_existing = np.sum(np.sum(hub_diffs, axis=1)) / self.m_constant

        new_dist_sum = current_dist_sum + dist_to_existing
        diversity_gain = (new_dist_sum - current_dist_sum) / self.num_pairs

        # Total gain calculation
        total_gain = (1 - self.lambda_param) * self.distortion * coverage_gain + (self.lambda_param * diversity_gain)

        return total_gain, (new_max_sims, new_dist_sum)