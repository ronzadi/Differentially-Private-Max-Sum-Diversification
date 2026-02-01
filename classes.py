from abc import ABC, abstractmethod
import numpy as np


class MSDObjective(ABC):
    """
    Abstract Base Class for submodular functions.
    """

    @abstractmethod
    def evaluate(self, S):
        """
        Calculates f(S).
        Returns: (value, auxiliary)
        """
        pass

    @abstractmethod
    def marginal_gain(self, e, S, current_val, auxiliary):
        """
        Calculates f(S U {e}) - f(S).
        Returns: (gain, new_auxiliary)
        """
        pass

    def add_one_element(self, e, S, current_val, auxiliary):
        """
        A default implementation for adding an element.
       use this to update state after a selection.
        """
        gain, new_auxiliary = self.marginal_gain(e, S, current_val, auxiliary)
        return current_val + gain, new_auxiliary


class MSDFacilityLocation(MSDObjective):
    def __init__(self, sim_matrix, facility_coords, lambda_param=1.0):
        """
        Args:
            sim_matrix: [Facilities x Customers]
            facility_coords: [Facilities x 2] (Lat/Lon)
            lambda_param: Weight for diversification (distance sum)
        """
        self.sim_matrix = sim_matrix
        self.coords = facility_coords
        self.lambda_param = lambda_param
        self.num_facilities, self.num_customers = sim_matrix.shape

    def evaluate(self, S):
        if not S:
            # Initial state: (max_sim_vector, running_dist_sum)
            return 0.0, (np.zeros(self.num_customers), 0.0)

        # Facility Location Part
        max_sims = np.max(self.sim_matrix[S, :], axis=0)
        fac_val = np.sum(max_sims)

        # Diversity Part (Pairwise sum)
        dist_sum = 0.0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                dist_sum += np.linalg.norm(self.coords[S[i]] - self.coords[S[j]])

        total_val = fac_val + (self.lambda_param * dist_sum)
        return total_val, (max_sims, dist_sum)

    def marginal_gain(self, e, S, current_val, auxiliary):
        """
        Incremental Gain: Delta_f = (Coverage_Gain) + lambda * (Dist_to_S)
        """
        # Unpack the current state
        max_sims, current_dist_sum = auxiliary

        # 1. Coverage Gain (Facility Location)
        new_max_sims = np.maximum(max_sims, self.sim_matrix[e, :])
        coverage_gain = np.sum(new_max_sims - max_sims)

        # 2. Diversity Gain (incremental distance)
        if not S:
            dist_to_existing = 0.0
        else:
            # Vectorized O(|S|) calculation
            diffs = self.coords[S] - self.coords[e]
            dist_to_existing = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

        total_gain = coverage_gain + (self.lambda_param * dist_to_existing)

        # Update the state forward
        new_dist_sum = current_dist_sum + dist_to_existing
        return total_gain, (new_max_sims, new_dist_sum)

    def add_one_element(self, e, S, current_val, auxiliary):
        """
        Commit e to S. The algorithms expect the new total value
        and the updated auxiliary state.
        """
        gain, updated_aux = self.marginal_gain(e, S, current_val, auxiliary)
        return current_val + gain, updated_aux