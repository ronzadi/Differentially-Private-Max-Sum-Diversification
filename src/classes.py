from abc import ABC, abstractmethod
import numpy as np


class GroundSet:
    def __init__(self, elements):
        self.elements = elements
        self.nb_elements = len(elements)

    def __iter__(self):
        return iter(self.elements)


class MSDObjective(ABC):

    @abstractmethod
    def evaluate(self, S, distort=False):
        pass

    @abstractmethod
    def marginal_gain(self, e, S, auxiliary, charge):
        pass

    def add_one_element(self, e, S, auxiliary):
        pass


class MSDUberObjective(MSDObjective):
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

        self.m_constant = 0.26

        self.num_passengers = len(passenger_coords)
        self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1

    def _get_similarity_row(self, grid_idx):
        diffs = np.abs(self.passengers - self.grid[grid_idx])
        d_ab = np.sum(diffs, axis=1)
        return 2.0 - (2.0 / (1.0 + np.exp(-200.0 * d_ab)))

    def evaluate(self, S, distort=True):
        self.num_queries += 1
        if not S:
            return 0.0, 0.0, 0.0, (np.zeros(self.num_passengers), 0.0)

        max_sims = np.zeros(self.num_passengers)
        for idx in S:
            max_sims = np.maximum(max_sims, self._get_similarity_row(idx))

        coverage_term = np.sum(max_sims) / self.num_passengers

        dist_sum = 0.0
        n_S = len(S)
        if n_S > 1:
            for i in range(n_S):
                for j in range(i + 1, n_S):
                    diff = np.abs(self.grid[S[i]] - self.grid[S[j]])
                    dist_sum += np.sum(diff) / self.m_constant

        avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        d_factor = self.distortion if distort else 1.0

        total_val = (1 - self.lambda_param) * d_factor * coverage_term + (self.lambda_param * avg_dist)

        return total_val, coverage_term, avg_dist, (max_sims, dist_sum)

    def marginal_gain(self, e, S, auxiliary, charge=True):

        if charge:
            self.num_queries += 1

        max_sims, current_dist_sum = auxiliary

        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)
        coverage_gain = (np.sum(new_max_sims) - np.sum(max_sims)) / self.num_passengers

        dist_to_existing = 0.0
        if S:
            hub_diffs = np.abs(self.grid[S] - self.grid[e])
            dist_to_existing = np.sum(np.sum(hub_diffs, axis=1)) / self.m_constant

        new_dist_sum = current_dist_sum + dist_to_existing
        diversity_gain = (new_dist_sum - current_dist_sum) / self.num_pairs

        total_gain = (1 - self.lambda_param) * self.distortion * coverage_gain + (self.lambda_param * diversity_gain)

        return total_gain, None

    def add_one_element(self, e, S, auxiliary):
        max_sims, current_dist_sum = auxiliary

        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)

        dist_to_existing = 0.0
        if S:
            hub_diffs = np.abs(self.grid[S] - self.grid[e])
            dist_to_existing = np.sum(np.sum(hub_diffs, axis=1)) / self.m_constant

        new_dist_sum = current_dist_sum + dist_to_existing

        return (new_max_sims, new_dist_sum)


class MSDAmazonObjective(MSDObjective):
    def __init__(self, reviews_df, product_categories, lambda_param, k, distortion, distance_matrix):
        unique_users = reviews_df['user_id'].unique()
        self.num_users = len(unique_users)

        self.ratings_lookup = (
            reviews_df.groupby('parent_asin')[['user_id', 'rating']]
            .apply(lambda x: list(map(tuple, x.values)))
            .to_dict()
        )
        self.user_sets = {
            asin: {u[0] for u in users}
            for asin, users in self.ratings_lookup.items()
        }
        self.categories = product_categories
        self.lambda_param = lambda_param
        self.k = k
        self.distortion = distortion
        self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1
        self.num_queries = 0
        self.distance_matrix = distance_matrix
        self.sensitivity = 1 / self.num_users

    def set_k(self, new_k):
        self.k = new_k
        self.num_pairs = (new_k * (new_k - 1)) / 2 if new_k > 1 else 1

    def _jaccard_distance(self, asin1, asin2):
        return self.distance_matrix[asin1][asin2]

    def marginal_gain(self, e, S, auxiliary, charge=True):
        if charge:
            self.num_queries += 1

        max_ratings, current_dist_sum = auxiliary

        relevance_diff = 0.0
        for u_id, rating in self.ratings_lookup.get(e, []):
            if u_id not in max_ratings or max_ratings[u_id] == 0:
                relevance_diff += 1.0

        relevance_gain = relevance_diff / self.num_users

        dist_to_existing = 0.0
        if S:
            for s_asin in S:
                dist_to_existing += self._jaccard_distance(e, s_asin)

        diversity_gain = (dist_to_existing / self.num_pairs) if self.num_pairs > 0 else 0.0

        total_gain = (1 - self.lambda_param) * self.distortion * relevance_gain + (self.lambda_param * diversity_gain)

        return total_gain, None

    def add_one_element(self, e_asin, S, auxiliary):
        max_ratings, current_dist_sum = auxiliary

        new_max_ratings = max_ratings.copy()

        for u_id, rating in self.ratings_lookup.get(e_asin, []):
            new_max_ratings[u_id] = new_max_ratings.get(u_id, 0) + 1

        dist_to_existing = 0.0
        for s_asin in S:
            dist_to_existing += self._jaccard_distance(e_asin, s_asin)

        return new_max_ratings, current_dist_sum + dist_to_existing

    def evaluate(self, S, distort=True):
        self.num_queries += 1
        if not S:
            return 0.0, 0.0, 0.0, ({}, 0.0)

        max_ratings = {}
        for asin in S:
            for u_id, rating in self.ratings_lookup.get(asin, []):
                max_ratings[u_id] = max_ratings.get(u_id, 0) + 1

        relevance_term = len(max_ratings) / self.num_users

        dist_sum = 0.0
        n_S = len(S)
        if n_S > 1:
            for i in range(n_S):
                for j in range(i + 1, n_S):
                    dist_sum += self._jaccard_distance(S[i], S[j])

        avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        d_factor = self.distortion if distort else 1
        total_val = (1 - self.lambda_param) * d_factor * relevance_term + (self.lambda_param * avg_dist)

        return total_val, relevance_term, avg_dist, (max_ratings, dist_sum)

    def evaluate_swap(self, e_out, e_in, S, auxiliary, distort=True):

        self.num_queries += 1
        coverage_counts, current_dist_sum = auxiliary
        users_of_out = self.user_sets.get(e_out, set())
        users_of_in = self.user_sets.get(e_in, set())

        current_users = set(coverage_counts.keys())
        lost_users = {u_id for u_id in users_of_out if coverage_counts.get(u_id) == 1}
        gained_users = {u_id for u_id in users_of_in if (u_id not in current_users or u_id in lost_users)}

        new_relevance = (len(coverage_counts) - len(lost_users) + len(gained_users)) / self.num_users

        dist_out = sum(self._jaccard_distance(e_out, s) for s in S if s != e_out)
        dist_in = sum(self._jaccard_distance(e_in, s) for s in S if s != e_out)
        new_dist_sum = current_dist_sum - dist_out + dist_in
        new_diversity = (new_dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        d_factor = self.distortion if distort else 1
        total_val = (1 - self.lambda_param) * d_factor * new_relevance + (self.lambda_param * new_diversity)
        return total_val

    def swap_element(self, e_out, e_in, S, auxiliary):

        max_ratings, current_dist_sum = auxiliary
        new_max_ratings = max_ratings.copy()

        for u_id, _ in self.ratings_lookup.get(e_out, []):
            new_max_ratings[u_id] -= 1
            if new_max_ratings[u_id] == 0:
                del new_max_ratings[u_id]

        for u_id, _ in self.ratings_lookup.get(e_in, []):
            new_max_ratings[u_id] = new_max_ratings.get(u_id, 0) + 1

        dist_out = sum(self._jaccard_distance(e_out, s) for s in S if s != e_out)
        dist_in = sum(self._jaccard_distance(e_in, s) for s in S if s != e_out)
        new_dist_sum = current_dist_sum - dist_out + dist_in

        return new_max_ratings, new_dist_sum


class MSDAmazonObjectiveMat(MSDObjective):
    def __init__(self, reviews_df, product_categories, lambda_param, k, distortion, distance_matrix):
        unique_users = reviews_df['user_id'].unique()
        self.num_users = len(unique_users)

        self.ratings_lookup = (
            reviews_df.groupby('parent_asin')[['user_id', 'rating']]
            .apply(lambda x: list(map(tuple, x.values)))
            .to_dict()
        )
        self.user_sets = {
            asin: {u[0] for u in users}
            for asin, users in self.ratings_lookup.items()
        }
        self.categories = product_categories
        self.lambda_param = lambda_param
        self.k = k
        self.distortion = distortion
        self.num_pairs = (k * (k - 1)) / 2 if k > 1 else 1
        self.num_queries = 0
        self.distance_matrix = distance_matrix
        self.sensitivity = 1 / self.num_users

    def set_k(self, new_k):
        self.k = new_k
        self.num_pairs = (new_k * (new_k - 1)) / 2 if new_k > 1 else 1

    def _jaccard_distance(self, asin1, asin2):
        return self.distance_matrix[asin1][asin2]

    def marginal_gain(self, e, S, auxiliary, charge=True):
        if charge:
            self.num_queries += 1
        max_ratings, _ = auxiliary

        relevance_diff = 0.0
        get_count = max_ratings.get
        for u_id, _ in self.ratings_lookup.get(e, []):
            if get_count(u_id, 0) == 0:
                relevance_diff += 1.0

        relevance_gain = relevance_diff / self.num_users

        dist_to_existing = 0.0
        if S:
            dist_map = self.distance_matrix[e]
            for s_asin in S:
                dist_to_existing += dist_map[s_asin]

        diversity_gain = (dist_to_existing / self.num_pairs) if self.num_pairs > 0 else 0.0
        return (1 - self.lambda_param) * self.distortion * relevance_gain + (self.lambda_param * diversity_gain), None

    def add_one_element(self, e_asin, S, auxiliary):
        max_ratings, current_dist_sum = auxiliary
        new_max_ratings = max_ratings.copy()

        for u_id, _ in self.ratings_lookup.get(e_asin, []):
            new_max_ratings[u_id] = new_max_ratings.get(u_id, 0) + 1

        dist_to_existing = 0.0
        if S:
            dist_map = self.distance_matrix[e_asin]
            for s_asin in S:
                dist_to_existing += dist_map[s_asin]

        return new_max_ratings, current_dist_sum + dist_to_existing

    def evaluate(self, S, distort=True):
        self.num_queries += 1
        if not S:
            return 0.0, 0.0, 0.0, ({}, 0.0)

        max_ratings = {}
        for asin in S:
            for u_id, _ in self.ratings_lookup.get(asin, []):
                max_ratings[u_id] = max_ratings.get(u_id, 0) + 1

        relevance_term = len(max_ratings) / self.num_users

        dist_sum = 0.0
        n_S = len(S)
        if n_S > 1:
            for i in range(n_S):
                asin_i = S[i]
                dist_map = self.distance_matrix[asin_i]
                for j in range(i + 1, n_S):
                    dist_sum += dist_map[S[j]]

        avg_dist = (dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0
        d_factor = self.distortion if distort else 1
        total_val = (1 - self.lambda_param) * d_factor * relevance_term + (self.lambda_param * avg_dist)

        return total_val, relevance_term, avg_dist, (max_ratings, dist_sum)

    def evaluate_swap(self, e_out, e_in, S, auxiliary, distort=True):
        self.num_queries += 1
        coverage_counts, current_dist_sum = auxiliary

        users_of_out = self.user_sets.get(e_out, set())
        users_of_in = self.user_sets.get(e_in, set())

        lost_count = 0
        get_count = coverage_counts.get
        for u_id in users_of_out:
            if get_count(u_id) == 1:
                lost_count += 1

        gained_count = 0
        for u_id in users_of_in:
            if u_id not in coverage_counts or (get_count(u_id) == 1 and u_id in users_of_out):
                gained_count += 1

        new_relevance = (len(coverage_counts) - lost_count + gained_count) / self.num_users

        dist_out = 0.0
        dist_in = 0.0
        dist_map_out = self.distance_matrix[e_out]
        dist_map_in = self.distance_matrix[e_in]

        for s in S:
            if s != e_out:
                dist_out += dist_map_out[s]
                dist_in += dist_map_in[s]

        new_dist_sum = current_dist_sum - dist_out + dist_in
        new_diversity = (new_dist_sum / self.num_pairs) if self.num_pairs > 0 else 0.0

        d_factor = self.distortion if distort else 1
        return (1 - self.lambda_param) * d_factor * new_relevance + (self.lambda_param * new_diversity)

    def swap_element(self, e_out, e_in, S, auxiliary):
        max_ratings, current_dist_sum = auxiliary
        new_max_ratings = max_ratings.copy()

        for u_id, _ in self.ratings_lookup.get(e_out, []):
            new_max_ratings[u_id] -= 1
            if new_max_ratings[u_id] == 0:
                del new_max_ratings[u_id]

        for u_id, _ in self.ratings_lookup.get(e_in, []):
            new_max_ratings[u_id] = new_max_ratings.get(u_id, 0) + 1

        dist_out = 0.0
        dist_in = 0.0
        dist_map_out = self.distance_matrix[e_out]
        dist_map_in = self.distance_matrix[e_in]
        for s in S:
            if s != e_out:
                dist_out += dist_map_out[s]
                dist_in += dist_map_in[s]

        return new_max_ratings, current_dist_sum - dist_out + dist_in