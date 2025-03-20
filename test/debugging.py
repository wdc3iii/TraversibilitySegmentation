import numpy as np

A_poly = np.load("/home/noelcs/hopper_ws/A_poly.npy")
b_poly = np.load("/home/noelcs/hopper_ws/b_poly.npy")
ineq = np.load("/home/noelcs/hopper_ws/ineq.npy")
A = np.load("/home/noelcs/hopper_ws/A.npy")
b = np.load("/home/noelcs/hopper_ws/b.npy")
v = np.load("/home/noelcs/hopper_ws/ext_pts.npy")

print(A_poly.shape, b_poly.shape, ineq.shape, A.shape, b.shape, v.shape)

print("Apoly")
print(A_poly)
print("bpoly")
print(b_poly)
print("ineq")
print(ineq)
print("A")
print(A)
print("b")
print(b)
print("ext")
print(v)

n = v.shape[0]
inds = np.array([np.arange(n), (np.arange(n) - 1) % n])  # Handles wrap-around for last element
sat_mat = A_poly @ v.T - b_poly[:, None]
sat_mask = np.all(np.abs(sat_mat[inds, :]) < 1e-12, axis=0)
ext_pts = np.repeat(v[None, :, :], n, axis=0)[sat_mask]

print('satmat')
print(sat_mat)
print('satmask')
print(sat_mask)
print('ext')
print(ext_pts)

