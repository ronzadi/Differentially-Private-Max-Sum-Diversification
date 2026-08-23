import numpy as np
import pandas as pd
import random
from scipy.spatial import ConvexHull


class UberOptimizer:
    def __init__(self, points, n_data):
        self.n_data = n_data

        self.points = np.asarray(points)
        if self.points.ndim == 1:
            self.points = self.points.reshape(-1, 2)

        self.hull = ConvexHull(self.points)
        self.A = self.hull.equations[:, :2]
        self.b = self.hull.equations[:, 2]

        self.norm = self._calculate_norm()

    def _calculate_norm(self):
        diffs = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        l1_dists = np.sum(np.abs(diffs), axis=-1)
        return np.max(l1_dists)

    def is_inside(self, pts_array, tol=1e-12):
        return np.all(self.A @ pts_array.T + self.b[:, None] <= tol, axis=0)

    def create_grid(self, n_locs, spurious):
        n_real = n_locs - spurious
        north_pole = self.points[np.argmax(self.points[:, 0])]

        if n_real <= 0:
            return np.tile(north_pole, (n_locs, 1))

        min_lat, min_lon = self.points.min(axis=0)
        max_lat, max_lon = self.points.max(axis=0)

        area_approx = (max_lat - min_lat) * (max_lon - min_lon)
        delta = np.sqrt(area_approx / (n_real * 2))

        real_pts = []
        for _ in range(3):
            lats = np.arange(min_lat, max_lat, delta)
            lons = np.arange(min_lon, max_lon, delta)
            lat_grid, lon_grid = np.meshgrid(lats, lons)
            candidates = np.c_[lat_grid.ravel(), lon_grid.ravel()]

            mask = self.is_inside(candidates)
            valid = candidates[mask]

            if len(valid) >= n_real:
                real_pts = valid
                break
            else:
                delta *= 0.8

        if len(real_pts) > n_real:
            indices = np.linspace(0, len(real_pts) - 1, n_real, dtype=int)
            real_pts = real_pts[indices]
        elif len(real_pts) < n_real:
            padding = n_real - len(real_pts)
            real_pts = np.vstack([real_pts, np.tile(north_pole, (padding, 1))])

        spurious_pts = np.tile(north_pole, (spurious, 1))
        return np.vstack([real_pts, spurious_pts])

    def process_raw_data(self, input_csv, output_csv):
        all_valid_points = []

        print("Filtering points inside hull...")
        for chunk in pd.read_csv(input_csv, chunksize=50000):
            chunk_pts = chunk.iloc[:, [0, 1]].values

            mask = self.is_inside(chunk_pts)
            valid_chunk = chunk_pts[mask]

            if len(valid_chunk) > 0:
                all_valid_points.append(valid_chunk)

        if not all_valid_points:
            raise ValueError("No points found inside the provided hull.")

        all_valid_points = np.vstack(all_valid_points)
        total_found = len(all_valid_points)
        print(f"Found {total_found} valid points in total.")

        if total_found >= self.n_data:
            indices = np.random.choice(total_found, int(self.n_data), replace=False)
            processed_data = all_valid_points[indices]
        else:
            print(f"Warning: Only {total_found} points found. Returning all.")
            processed_data = all_valid_points

        df_out = pd.DataFrame(processed_data, columns=['lat', 'lon'])
        df_out.to_csv(output_csv, index=False)

        return df_out.values

    def read_from_file(self, output_csv):

        df = pd.read_csv(output_csv)
        return df.values

    def read_sample_from_file(self, output_csv, pct):

        df = pd.read_csv(output_csv)
        df = df.sample(frac=pct / 100)
        return df.values

    def evaluate_function(self, S_indices, grid_coords, passenger_coords):
        if not S_indices:
            return 0.0

        selected_hubs = grid_coords[list(S_indices)]
        sum_dist = 0

        for p in passenger_coords:
            dists = np.sum(np.abs(selected_hubs - p), axis=1) / self.norm
            sum_dist += np.min(dists)

        return self.n_data - sum_dist


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

    manhattan_grid = opt.create_grid(n_locs=1000)
    print(f"Generated {len(manhattan_grid)} points inside Manhattan.")