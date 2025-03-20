import numpy as np
import cdd
from scipy.spatial import ConvexHull
import time

# Define points
# points = np.array([
#     [1, -1],  # Vertex 0
#     [1, 1],   # Vertex 1
#     [-1, 1],  # Vertex 2
#     [-1, -1]  # Vertex 3
# ])
points = np.random.normal(size=(1000, 2))

# Convert to homogeneous coordinates for cddlib
points_h = np.hstack((np.ones((points.shape[0], 1)), points))

# Number of trials
num_trials = 10

# Time pycddlib
pycddlib_times = []
for _ in range(num_trials):
    start = time.time()
    mat = cdd.Matrix(points_h.tolist(), number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    h_rep = np.array(poly.get_inequalities())
    pycddlib_times.append(time.time() - start)

# Compute average pycddlib time
avg_pycddlib_time = np.mean(pycddlib_times)

print("H-representation from pycddlib (-b, A):")
print(h_rep)

# Time SciPy ConvexHull
scipy_times = []
for _ in range(num_trials):
    start = time.time()
    hull = ConvexHull(points)
    A = hull.equations[:, :-1]  # Normal vectors
    b = -hull.equations[:, -1]  # Offsets
    scipy_times.append(time.time() - start)

# Compute average SciPy ConvexHull time
avg_scipy_time = np.mean(scipy_times)

print("\nH-representation from SciPy ConvexHull (A, b):")
for i in range(len(A)):
    print(f"{A[i]} <= {b[i]}")

print(f"\nAverage execution time over {num_trials} runs:")
print(f"pycddlib: {avg_pycddlib_time:.6f} seconds")
print(f"SciPy ConvexHull: {avg_scipy_time:.6f} seconds")

