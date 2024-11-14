import itertools

import haversine
import numpy as np


# Cities are stored in (lon, lat) format

NEW_YORK = (-74.0060, 40.7128)
PARIS = (2.3522, 48.8566)
SYDNEY = (151.2093, -33.8688)

cities = [NEW_YORK, PARIS, SYDNEY]

# Get all pairs of city coordinates in (x1, y1, x2, y2) format

pairs = [(*a, *b) for a, b in itertools.product(cities, repeat=2)]
x1, y1, x2, y2 = [np.array(n) for n in zip(*pairs)]

# Allocate space for the output and run the kernel

distances = np.empty(len(pairs))
haversine.distance(len(pairs), x1, y1, x2, y2, distances)

# Verify that the results make sense

print(f"{distances=}")

assert np.allclose(
	distances,
	[
		0,
		5.83724090e03,
		1.59887555e04,
		5.83724090e03,
		0.0,
		1.69604974e04,
		1.59887555e04,
		1.69604974e04,
		0.0,
	],
)

