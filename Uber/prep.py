import numpy as np
import pandas as pd
import random
import time


class UberOptimizer:
    def __init__(self, box, n_data=10000):
        """
        box: [latNorth, latSouth, latEast, latWest, lonNorth, lonSouth, lonEast, lonWest]
        """
        self.box = box
        self.n_data = n_data

        # Unpack for coordinate math
        self.lat_n, self.lat_s, self.lat_e, self.lat_w = box[0:4]
        self.lon_n, self.lon_s, self.lon_e, self.lon_w = box[4:8]

        max_val = max(
            abs(self.lon_n - self.lon_s) + abs(self.lat_n - self.lat_s),
            abs(self.lon_e - self.lon_w) + abs(self.lat_e - self.lat_w),
            abs(self.lon_n - self.lon_w) + abs(self.lat_n - self.lat_w),
            abs(self.lon_e - self.lon_s) + abs(self.lat_e - self.lat_s)
        )
        self.norm = max_val

    def create_grid(self, n_locs, n_cols, spurious=0):
        """Equivalent to createGrid in the C++ code."""
        n_main = n_locs - spurious
        n_rows = n_main // n_cols

        coords = []
        for i in range(n_main):
            row_idx = i // n_cols
            col_idx = i % n_cols

            # Step progress (0.0 to 1.0)
            step_v = row_idx * (1.0 / (n_rows - 1)) if n_rows > 1 else 0
            step_h = col_idx * (1.0 / (n_cols - 1)) if n_cols > 1 else 0

            # Vector addition to create the tilted parallelogram
            lat = self.lat_s + (self.lat_w - self.lat_s) * step_v + (self.lat_e - self.lat_s) * step_h
            lon = self.lon_s + (self.lon_w - self.lon_s) * step_v + (self.lon_e - self.lon_s) * step_h
            coords.append([lat, lon])

        # Add spurious points at the North pole
        for _ in range(spurious):
            coords.append([self.lat_n, self.lon_n])

        return np.array(coords)

    # def process_raw_data(self, input_csv, output_csv):
    #     """Equivalent to dataInit (Pre-processing with Tilted Box and Reservoir Sampling)"""
    #     # Line slopes for the quadrilateral check
    #     m1 = (self.lat_s - self.lat_w) / (self.lon_s - self.lon_w)
    #     m2 = (self.lat_e - self.lat_s) / (self.lon_e - self.lon_s)
    #     m3 = (self.lat_n - self.lat_w) / (self.lon_n - self.lon_w)
    #     m4 = (self.lat_e - self.lat_n) / (self.lon_e - self.lon_n)
    #
    #     processed_data = []
    #     count = 0
    #
    #     # Reading in chunks to handle large file sizes without crashing RAM
    #     for chunk in pd.read_csv(input_csv, chunksize=10000):
    #         # C++ Column Mapping: 0: Date, 1: Lat, 2: Lon
    #         lats = chunk.iloc[:, 1].values
    #         lons = chunk.iloc[:, 2].values
    #
    #         # Perform the 4-check quadrilateral filter
    #         check1 = lats > self.lat_w + m1 * (lons - self.lon_w)
    #         check2 = lats > self.lat_s + m2 * (lons - self.lon_s)
    #         check3 = lats < self.lat_w + m3 * (lons - self.lon_w)
    #         check4 = lats < self.lat_n + m4 * (lons - self.lon_n)
    #
    #         valid_mask = check1 & check2 & check3 & check4
    #         valid_points = np.column_stack((lats[valid_mask], lons[valid_mask]))
    #
    #         for pt in valid_points:
    #             if len(processed_data) < self.n_data:
    #                 processed_data.append(pt)
    #             else:
    #                 # Reservoir Sampling logic
    #                 prob = self.n_data / (count + 1)
    #                 if random.random() < prob:
    #                     idx = random.randint(0, self.n_data - 1)
    #                     processed_data[idx] = pt
    #             count += 1
    #
    #     # Shuffle and Save
    #     random.shuffle(processed_data)
    #     df_out = pd.DataFrame(processed_data, columns=['lat', 'lon'])
    #     df_out.to_csv(output_csv, index=False)
    #     return df_out.values

    def process_raw_data(self, input_csv, output_csv):
        # Define the 4 corners in order to form a perimeter
        pts = [
            (self.lat_s, self.lon_s),
            (self.lat_w, self.lon_w),
            (self.lat_n, self.lon_n),
            (self.lat_e, self.lon_e)
        ]

        processed_data = []
        count = 0

        for chunk in pd.read_csv(input_csv, chunksize=10000):
            lats = chunk.iloc[:, 1].values
            lons = chunk.iloc[:, 2].values

            # Cross product check: (p2.y-p1.y)*(px-p1.x) - (p2.x-p1.x)*(py-p1.y)
            def side_check(p1, p2, px, py):
                return (p2[1] - p1[1]) * (px - p1[0]) - (p2[0] - p1[0]) * (py - p1[1])

            s1 = side_check(pts[0], pts[1], lats, lons)
            s2 = side_check(pts[1], pts[2], lats, lons)
            s3 = side_check(pts[2], pts[3], lats, lons)
            s4 = side_check(pts[3], pts[0], lats, lons)

            # Point is inside if it's on the same side of all four boundary vectors
            valid_mask = ((s1 > 0) & (s2 > 0) & (s3 > 0) & (s4 > 0)) | \
                         ((s1 < 0) & (s2 < 0) & (s3 < 0) & (s4 < 0))

            valid_points = np.column_stack((lats[valid_mask], lons[valid_mask]))

            for pt in valid_points:
                if len(processed_data) < self.n_data:
                    processed_data.append(pt)
                else:
                    prob = self.n_data / (count + 1)
                    if random.random() < prob:
                        idx = random.randint(0, self.n_data - 1)
                        processed_data[idx] = pt
                count += 1

        random.shuffle(processed_data)
        df_out = pd.DataFrame(processed_data, columns=['lat', 'lon'])
        df_out.to_csv(output_csv, index=False)
        return df_out.values
    def evaluate_function(self, S_indices, grid_coords, passenger_coords):
        """Equivalent to funValue (The Facility Location Objective)"""
        if not S_indices:
            return 0.0

        selected_hubs = grid_coords[list(S_indices)]

        # Calculate Manhattan distance for every passenger to every selected hub
        # Result is [N_passengers x N_selected_hubs]
        sum_dist = 0
        for p in passenger_coords:
            # Vectorized Manhattan distance calculation
            dists = np.sum(np.abs(selected_hubs - p), axis=1) / self.norm
            sum_dist += np.min(dists)

        return self.n_data - sum_dist

if __name__ == '__main__':
    # --- Example Usage ---
    manhattan_box = [40.81794, 40.6866, 40.80204, 40.71315,
                     -73.96483, -73.99197, -73.91436, -74.04519]

    opt = UberOptimizer(manhattan_box)

    # 1. Create the tilted grid (1000 locations, 20 columns)
    grid = opt.create_grid(1000, 20)

    print(grid)
    # 2. Filter and sample raw Uber data
    # passengers = opt.process_raw_data("uber-raw-data-apr14.csv", "uber-small.csv")

    # 3. Dummy Evaluation (Example)
    # val = opt.evaluate_function({0, 10, 50}, grid, passengers)