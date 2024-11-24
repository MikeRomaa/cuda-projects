import cudf
import haversine
import numpy as np


taxi = cudf.read_parquet("/data/csc59866_f24/tlcdata/yellow_tripdata_2009-01.parquet")

x1 = taxi['Start_Lon'].to_numpy()
y1 = taxi['Start_Lat'].to_numpy()
x2 = taxi['End_Lon'].to_numpy()
y2 = taxi['End_Lat'].to_numpy()

distances = np.zeros(len(x1))
haversine.distance(len(x1), x1, y1, x2, y2, distances)

print(distances)

