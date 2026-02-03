import numpy as np
import pandas as pd
import random
from scipy.spatial import ConvexHull


class UberOptimizer:
    def __init__(self, points, n_data):
        """
        Initializes the optimizer with a Convex Hull boundary.
        points: A list of (lat, lon) tuples or a 2D numpy array.
        """
        self.n_data = n_data

        # 1. Ensure points are a 2D numpy array (N, 2)
        self.points = np.asarray(points)
        if self.points.ndim == 1:
            # Safety for flat lists
            self.points = self.points.reshape(-1, 2)

        # 2. Build the Convex Hull and extract the Half-space equations
        # Equations are in the form: a*lat + b*lon + c <= 0
        self.hull = ConvexHull(self.points)
        self.A = self.hull.equations[:, :2]  # Coefficients [a, b]
        self.b = self.hull.equations[:, 2]  # Constant [c]

        # 3. Calculate L1 Normalization (Diameter of Manhattan)
        # This is the max possible distance between any two points in the hull
        self.norm = self._calculate_norm()

    def _calculate_norm(self):
        """Vectorized L1 diameter calculation."""
        # Calculate all-to-all L1 distances between hull vertices
        diffs = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        l1_dists = np.sum(np.abs(diffs), axis=-1)
        return np.max(l1_dists)

    def is_inside(self, pts_array, tol=1e-12):
        """
        Vectorized check: Is point(s) inside the hull?
        pts_array: numpy array of shape (N, 2)
        Returns: Boolean mask of shape (N,)
        """
        # Matrix multiplication check: A @ x + b <= 0
        # We transpose pts_array to (2, N) to align with A (Facets, 2)
        return np.all(self.A @ pts_array.T + self.b[:, None] <= tol, axis=0)

    def create_grid(self, n_locs):
        """
        Generates a candidate grid clipped to the Manhattan hull.
        """
        min_lat, min_lon = self.points.min(axis=0)
        max_lat, max_lon = self.points.max(axis=0)

        # Create a bounding box slightly denser than n_locs to account for clipping
        grid_size = int(np.sqrt(n_locs * 2))
        lats = np.linspace(min_lat, max_lat, grid_size)
        lons = np.linspace(min_lon, max_lon, grid_size)

        lat_grid, lon_grid = np.meshgrid(lats, lons)
        candidate_pts = np.c_[lat_grid.ravel(), lon_grid.ravel()]

        # Clip to the real Manhattan shape
        mask = self.is_inside(candidate_pts)
        return candidate_pts[mask]

    def process_raw_data(self, input_csv, output_csv):
        """
        Filters Uber data using the Convex Hull and Reservoir Sampling.
        """
        processed_data = []
        count = 0

        # Chunked reading to save RAM
        for chunk in pd.read_csv(input_csv, chunksize=50000):
            # Assume Col 1: Lat, Col 2: Lon (Typical for Uber Raw Data)
            chunk_pts = chunk.iloc[:, [1, 2]].values

            # Vectorized filter check
            mask = self.is_inside(chunk_pts)
            valid_points = chunk_pts[mask]

            for pt in valid_points:
                if len(processed_data) < self.n_data:
                    processed_data.append(pt)
                else:
                    # Reservoir Sampling logic
                    prob = self.n_data / (count + 1)
                    if random.random() < prob:
                        idx = random.randint(0, self.n_data - 1)
                        processed_data[idx] = pt
                count += 1

        # Save the filtered, sampled data
        random.shuffle(processed_data)
        df_out = pd.DataFrame(processed_data, columns=['lat', 'lon'])
        df_out.to_csv(output_csv, index=False)
        return df_out.values

    def evaluate_function(self, S_indices, grid_coords, passenger_coords):
        """
        Facility Location Objective: Sum(1 - dist_min/norm)
        """
        if not S_indices:
            return 0.0

        selected_hubs = grid_coords[list(S_indices)]
        sum_dist = 0

        # Calculate min distance for each passenger to the set of hubs
        for p in passenger_coords:
            # Manhattan distance normalized by Manhattan's diameter
            dists = np.sum(np.abs(selected_hubs - p), axis=1) / self.norm
            sum_dist += np.min(dists)

        return self.n_data - sum_dist


# --- Usage Example ---
if __name__ == "__main__":
    full_island_hull = [
        (40.7005038, -74.0144209), (40.7112088, -73.9776851),
        (40.7282434, -73.9720702), (40.7418214, -73.9733576),
        (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482),
        (40.8769142, -73.9269985), (40.8512745, -73.9448513),
        (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    opt = UberOptimizer(full_island_hull, n_data=20000)

    # Generate the grid clipped to Manhattan
    manhattan_grid = opt.create_grid(n_locs=1000)
    print(f"Generated {len(manhattan_grid)} points inside Manhattan.")