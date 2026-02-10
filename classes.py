from abc import ABC, abstractmethod
from collections import defaultdict

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
    def evaluate(self, S, distort=False):
        """
        Calculates f(S).
        Returns: (value, auxiliary)
        """
        pass

    @abstractmethod
    def marginal_gain(self, e, S, auxiliary, charge):
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



class MSDUberObjective(MSDObjective):  # Inherits from MSDObjective if required by your script
    def __init__(self, passenger_coords, grid_coords, lambda_param, k, distortion, sensitivity):
        """
        Args:
            passenger_coords: [N x 2] array of Uber pickup locations
            grid_coords: [M x 2] array of candidate hub locations
            lambda_param: Weighting between coverage and diversity
            k: Cardinality constraint (total hubs to select)
            distortion: Multiplier for the submodular term
        """
        self.passengers = passenger_coords
        self.grid = grid_coords
        self.lambda_param = lambda_param
        self.k = k
        self.distortion = distortion
        self.sensitivity = sensitivity
        self.num_queries = 0

        # Hardcoded normalization constant
        self.m_constant = 0.26

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
        self.num_queries += 1
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

        return total_val, coverage_term, avg_dist, (max_sims, dist_sum)

    def marginal_gain(self, e, S, auxiliary, charge=True):
        """
        Efficiently calculates the increase in utility if element 'e' is added to 'S'.
        """
        if charge:
            self.num_queries += 2

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



class MSDAmazonObjective(MSDObjective):
    def __init__(self, reviews_df, product_categories, lambda_param, k, distortion):
        # We still need unique_users to calculate the average (N) accurately
        unique_users = reviews_df['user_id'].unique()
        self.num_users = len(unique_users)

        # Group ratings by product: {asin: [(user_id, rating), ...]}
        # Note: We store the actual user_id now, no need for an integer map
        self.ratings_lookup = defaultdict(list)
        for _, row in reviews_df.iterrows():
            self.ratings_lookup[row['parent_asin']].append((row['user_id'], row['rating']))

        self.categories = product_categories  # Dict: asin -> set()
        self.lambda_param = lambda_param
        self.k = k
        self.distortion = distortion
        self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1
        self.num_queries = 0

        # Sensitivity based on max rating (usually 5) / N
        # If your ratings are normalized to [0,1], use 1/N
        self.sensitivity = 5.0 / self.num_users

    def _jaccard_distance(self, asin1, asin2):
        set1 = self.categories.get(asin1, set())
        set2 = self.categories.get(asin2, set())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1.0 - (intersection / union) if union > 0 else 1.0

    def marginal_gain(self, e, S, auxiliary, charge=True):
        if charge:
            self.num_queries += 1

        # Unpack current state: max_ratings is now a DICT {user_id: current_max}
        max_ratings, current_dist_sum = auxiliary

        # 1. Relevance Gain
        relevance_diff = 0.0
        for u_id, rating in self.ratings_lookup.get(e, []):
            # If user not in dict, their current max is 0
            current_u_max = max_ratings.get(u_id, 0.0)
            if rating > current_u_max:
                relevance_diff += (rating - current_u_max)

        relevance_gain = relevance_diff / self.num_users

        # 2. Diversity Gain
        dist_to_existing = 0.0
        if S:
            for s_asin in S:
                dist_to_existing += self._jaccard_distance(e, s_asin)

        diversity_gain = (dist_to_existing / self.num_pairs) if self.num_pairs > 0 else 0.0

        total_gain = (1 - self.lambda_param) * self.distortion * relevance_gain + (self.lambda_param * diversity_gain)

        return total_gain, None

    def add_one_element(self, e_asin, S, auxiliary):
        max_ratings, current_dist_sum = auxiliary

        # Shallow copy of the dict is enough since values are floats
        new_max_ratings = max_ratings.copy()

        for u_id, rating in self.ratings_lookup.get(e_asin, []):
            current_u_max = new_max_ratings.get(u_id, 0.0)
            if rating > current_u_max:
                new_max_ratings[u_id] = rating

        dist_to_existing = 0.0
        for s_asin in S:
            dist_to_existing += self._jaccard_distance(e_asin, s_asin)

        return new_max_ratings, current_dist_sum + dist_to_existing

    def evaluate(self, S, distort=True):
        self.num_queries += 1
        if not S:
            return 0.0, 0.0, 0.0, ({}, 0.0)  # Empty dict instead of zero array

        max_ratings = {}
        for asin in S:
            for u_id, rating in self.ratings_lookup.get(asin, []):
                if rating > max_ratings.get(u_id, 0.0):
                    max_ratings[u_id] = rating

        # relevance = sum(max_ratings) / N. Users not in dict contribute 0 to sum.
        relevance_term = sum(max_ratings.values()) / self.num_users

        dist_sum = 0.0
        n_S = len(S)
        if n_S > 1:
            for i in range(n_S):
                for j in range(i + 1, n_S):
                    dist_sum += self._jaccard_distance(S[i], S[j])

        avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        d_factor = self.distortion if distort else 1
        total_val = (1 - self.lambda_param) * d_factor * relevance_term + (self.lambda_param * avg_dist)

        return total_val, relevance_term, avg_dist,  (max_ratings, dist_sum)
#
# class MSDAmazonObjective(MSDObjective):
#     def __init__(self, reviews_df, product_categories, lambda_param, k, distortion):
#         """
#         Args:
#             reviews_df: DataFrame with ['user_id', 'parent_asin', 'rating']
#             product_categories: Dict mapping parent_asin -> set of category strings
#         """
#         # Map user_ids to integers for efficient array indexing in auxiliary state
#         unique_users = reviews_df['user_id'].unique()
#         self.user_map = {user: i for i, user in enumerate(unique_users)}
#         self.num_users = len(unique_users)
#
#         # Group ratings by product for fast lookups during marginal gain
#         # {product_asin: [(user_idx, rating), ...]}
#         self.ratings_lookup = defaultdict(list)
#         for _, row in reviews_df.iterrows():
#             u_idx = self.user_map[row['user_id']]
#             self.ratings_lookup[row['parent_asin']].append((u_idx, row['rating']))
#
#         self.categories = product_categories  # Dict: asin -> set()
#         self.lambda_param = lambda_param
#         self.k = k
#         self.distortion = distortion
#         self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1
#         self.num_queries = 0
#         self.sensitivity = 1/self.num_users
#
#     def _jaccard_distance(self, asin1, asin2):
#         set1 = self.categories.get(asin1, set())
#         set2 = self.categories.get(asin2, set())
#         intersection = len(set1.intersection(set2))
#         union = len(set1.union(set2))
#         return 1.0 - (intersection / union) if union > 0 else 1.0
#
#     def marginal_gain(self, e, S, auxiliary, charge=True):
#         if charge:
#             self.num_queries += 1
#
#         # Unpack the current state (read-only)
#         max_ratings, current_dist_sum = auxiliary
#
#         # 1. Relevance Gain (Zero Copies)
#         relevance_diff = 0.0
#         # Only look at the specific users who rated product 'e'
#         for u_idx, rating in self.ratings_lookup.get(e, []):
#             # If this product is better than the user's current favorite in S...
#             if rating > max_ratings[u_idx]:
#                 # ...add only the improvement to the total gain
#                 relevance_diff += (rating - max_ratings[u_idx])
#
#         relevance_gain = relevance_diff / self.num_users
#
#         # 2. Diversity Gain (Calculated on the fly)
#         dist_to_existing = 0.0
#         if S:
#             for s_asin in S:
#                 dist_to_existing += self._jaccard_distance(e, s_asin)
#
#         diversity_gain = (dist_to_existing / self.num_pairs) if self.num_pairs > 0 else 0.0
#
#         total_gain = (1 - self.lambda_param) * self.distortion * relevance_gain + (self.lambda_param * diversity_gain)
#
#         # We return NONE for the auxiliary because we didn't waste time/RAM building it.
#         # This prevents the "NoneType" error ONLY IF the greedy loop
#         # uses add_one_element to do the final update.
#         return total_gain, None
#
#     def add_one_element(self, e_asin, S, auxiliary):
#         """
#         Update the auxiliary state (the max_ratings array) when an element is chosen.
#         """
#         max_ratings, current_dist_sum = auxiliary
#         new_max_ratings = max_ratings.copy()
#
#         dist_to_existing = 0.0
#         for u_idx, rating in self.ratings_lookup.get(e_asin, []):
#             if rating > new_max_ratings[u_idx]:
#                 new_max_ratings[u_idx] = rating
#
#         for s_asin in S:
#             dist_to_existing += self._jaccard_distance(e_asin, s_asin)
#
#         return new_max_ratings, current_dist_sum + dist_to_existing
#
#     def evaluate(self, S, distort=True):
#         """
#         Calculates f(S) from scratch.
#         Returns: (value, auxiliary) where auxiliary is (max_ratings_vec, current_dist_sum)
#         """
#         self.num_queries += 1
#         if not S:
#             return 0.0, (np.zeros(self.num_users), 0.0)
#
#         # 1. Relevance: avg over users of max ratings
#         max_ratings = np.zeros(self.num_users)
#         for asin in S:
#             for u_idx, rating in self.ratings_lookup.get(asin, []):
#                 if rating > max_ratings[u_idx]:
#                     max_ratings[u_idx] = rating
#         relevance_term = np.mean(max_ratings)
#
#         # 2. Diversity: Sum of pairwise distances
#         dist_sum = 0.0
#         n_S = len(S)
#         if n_S > 1:
#             for i in range(n_S):
#                 for j in range(i + 1, n_S):
#                     dist_sum += self._jaccard_distance(S[i], S[j])
#
#         avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0
#
#         distort = self.distortion if distort else 1
#         total_val = (1 - self.lambda_param) * self.distortion * relevance_term + (self.lambda_param * avg_dist)
#         return total_val, (max_ratings, dist_sum)