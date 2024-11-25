import math
import time
from contextlib import contextmanager

import cudf
import haversine
import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS_KM = 6371.0088


@contextmanager
def _measure_perf(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()

    print(f"{label}: {(end - start):.10f} s")


def _haversine(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the great-circle distance between two points on the globe."""
    phi1 = math.radians(y1)
    phi2 = math.radians(y2)
    lambda1 = math.radians(x1)
    lambda2 = math.radians(x2)

    d_phi = phi2 - phi1
    d_lambda = lambda2 - lambda1

    hav = (1 - math.cos(d_phi) + math.cos(phi1) * math.cos(phi2) * (1 - math.cos(d_lambda))) / 2

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


if __name__ == "__main__":
    # Load data from files

    with _measure_perf("Loading dataset"):
        df = cudf.concat(
            [
                cudf
                .read_parquet(f"/data/csc59866_f24/tlcdata/yellow_tripdata_2009-{month:>02}.parquet")
                .query(
                    (
                        '-74.15 <= Start_Lon <= -73.7004'
                        ' and -74.15 <= End_Lon <= -73.7004'
                        ' and 40.5774 <= Start_Lat <= 40.9176'
                        ' and 40.5774 <= End_Lat <= 40.9176'
                    )
                )
                [["Start_Lon", "Start_Lat", "End_Lon", "End_Lat"]]
                for month in range(1, 13)
            ]
        )

    x1 = df['Start_Lon'].to_numpy()
    y1 = df['Start_Lat'].to_numpy()
    x2 = df['End_Lon'].to_numpy()
    y2 = df['End_Lat'].to_numpy()

    # Get distances for each trip

    distances = np.zeros(len(x1))

    with _measure_perf("Haversine on GPU"):
        haversine.distance(len(x1), x1, y1, x2, y2, distances)

    with _measure_perf("Haversine on CPU"):
        for args in zip(x1, y1, x2, y2):
            _haversine(*args)

    # (Bonus 5) Create plots for the data

    fig = plt.figure()
    start_plt = plt.subplot(221, title="Trip Start Locations", xlabel="Longitude", ylabel="Latitude")
    end_plt = plt.subplot(222, title="Trip End Locations", xlabel="Longitude", ylabel="Latitude")
    dist_plt = plt.subplot(212, title="Trip Distances", xlabel="Distance (km)", ylabel="No. Trips")

    bounds = [[-74.15, -73.7004], [40.5774, 40.9176]]

    start_plt.hist2d(x1, y1, bins=100, range=bounds, cmap="binary")
    end_plt.hist2d(x2, y2, bins=100, range=bounds, cmap="binary")
    dist_plt.hist(distances, bins=50)

    plt.tight_layout()
    plt.savefig("plot.png")

    # We tried to complete bonus 6 as well, but loading the coordinates as well as the datetime would
    # result in too much memory being allocated on-device. Attempting to load the dataset into host
    # memory and then selectively loading part of it onto the device also failed.

