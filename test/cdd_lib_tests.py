import numpy as np
import cdd
import time

# Define points
points = np.array([
    [1, -1],  # Vertex 0
    [1, 0],   # Vertex 1
    [0, 0.9],
    [-1, 0],  # Vertex 2
    [-1, -1]  # Vertex 3
])

A = np.array([
    [0, -1],
    [1, 0],
    [1, 1],
    [-1, 1],
    [-1, 0],
    [0, 1]
])
b = np.array([
    1,
    1,
    1,
    1,
    1,
    1.1
])
print(A); print(b); print('\n')


mat = cdd.Matrix(np.hstack([b[:, None], -A]).tolist(), number_type='float')
mat.rep_type = cdd.RepType.INEQUALITY
mat.canonicalize()
poly = cdd.Polyhedron(mat)
ext_pts = np.array(poly.get_generators())[:, 1:]
ineq = np.array(poly.get_inequalities())
# ineq /= np.linalg.norm(ineq, axis=-1, keepdims=True)  # normalize hyperplane representation
A_poly, b_poly = -ineq[:, 1:], ineq[:, 0]
norm_A = np.linalg.norm(A_poly, axis=-1, keepdims=True)
A_poly /= norm_A
b_poly /= norm_A.squeeze()
print(A_poly); print(b_poly); print('\n')
print(ext_pts); print('\n')

n = ext_pts.shape[0]
inds = np.array([np.arange(n), (np.arange(n) - 1) % n])  # Handles wrap-around for last element

# Shape: (2, num_constraints, num_points)
sat_mat = A_poly @ ext_pts.T - b_poly[:, None]
sat_mask = np.all(sat_mat[inds, :] == 0, axis=0)
ext_pts = np.repeat(ext_pts[None, :, :], n, axis=0)[sat_mask]
print(sat_mat); print(sat_mask); print(ext_pts)