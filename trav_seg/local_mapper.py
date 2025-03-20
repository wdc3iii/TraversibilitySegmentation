import random
import threading
import numpy as np
from sklearn.cluster import KMeans
from trav_seg.trav_segmenter import TravSegmenter
from scipy.spatial import ConvexHull, HalfspaceIntersection
import cdd


class LocalMapper:

    FREE = 0
    OCC = 100
    UNKNOWN = -1

    def __init__(self, disc: float, dim: int, K: int, initial_free_radius: float, recenter_thresh: float,
                 buffer_mult: float=1.2, max_eig: float=1, min_keep_pts: int=10, local_prompt: bool=False):
        """nitializes a LocalMapper object

        Args:
            disc (float): spatial discretization in meters
            dim (int): number of grid cells in the map
            K (int): number of free space regions to identify
            init_free_radius (float): radius around the robot to initially assume is free
            recenter_thresh (float): Distance from the center of the map the robot can move before recentering
            buffer_mult (float, optional): Proportion of the best-fit ellipse to use. Defaults to 1.2.
            max_eig (float, optional): Minimum size ellipse to consider. Defaults to 1.
        """
        # Occupany grid
        self.occ_grid = np.ones((dim, dim), dtype=int) * self.UNKNOWN
        x_locs = np.repeat(np.arange(self.occ_grid.shape[1])[None, :], self.occ_grid.shape[0], axis=0) * disc + disc / 2
        y_locs = np.repeat(np.arange(self.occ_grid.shape[0])[:, None], self.occ_grid.shape[1], axis=1) * disc + disc / 2
        self.locs = np.hstack((x_locs.flatten()[:, None], y_locs.flatten()[:, None]))
        self.disc = disc
        self.recenter_thresh = recenter_thresh
        mid = dim // 2
        dx = int(initial_free_radius // disc)
        self.occ_grid[mid-dx:mid+dx, mid-dx:mid+dx] = self.FREE
        # Assumes hopper gets dropped at (or around) [0, 0]
        self.map_origin = np.array([-mid *  self.disc, -mid * self.disc])

        # Free-space ID parameters
        self.buffer_mult = buffer_mult
        self.max_eig = max_eig
        self.min_keep_pts = min_keep_pts
        self.local_prompt = local_prompt
        self.K = K
        self.rng = random.Random(42)  # Seed for reproducibility
        self.kmeans = None
        self.labels = None
        self.free_mask = None

        # Segmenter
        self.trav_seg = TravSegmenter(record=False, o3d_vis=False, print_timing=False, local_prompt=self.local_prompt)

        # Threading locks
        self.trav_seg_lock = threading.Lock()
        self.occ_grid_lock = threading.Lock()
        
        if self.local_prompt:
            self.prompt_seg()

    @property
    def map_center(self):
        return self.map_origin + np.array([self.occ_grid.shape[1], self.occ_grid.shape[0]]) // 2 * self.disc
    
    def prompt_seg(self):
        """Runs local prompting for segmenter"""
        self.trav_seg.capture_frame()
        self.trav_seg.prompt_seg()

    def capture_frame(self, p, R):
        """Captures a frame from the segmenter

        Args:
            p (np.ndarray): position of the camera in 3D space
            R (np.ndarray): orientation of the camera in 3D space
        """
        with self.trav_seg_lock:
            self.trav_seg.capture_frame()
            self.trav_seg.transform_point_cloud_(p, R)

    def segment_frame(self):
        """Segment the current frame
        """
        with self.trav_seg_lock:
            self.trav_seg.segment_frame()

    def reset_first_frame(self):
        """Reset the segmenter first frame
        """
        with self.trav_seg_lock:
            self.trav_seg.reset_first_frame()

    def fit_free_space(self):
        """Fits a set of K convex polytopes to the free space
        """
        # Separate free and occupied points
        with self.occ_grid_lock:
            self.free_mask = self.occ_grid.flatten() == self.FREE
        free_pts = self.locs[self.free_mask, :] + self.map_origin
        # Run kmeans
        self.kmeans = KMeans(
            n_clusters=self.K,
            n_init="auto",
            random_state=self.rng.randint(0, 100000),
            init=self.kmeans.cluster_centers_ if self.kmeans is not None else 'k-means++',
            tol=self.disc / 5
        )
        self.labels = self.kmeans.fit_predict(free_pts)
    
        # Loop over polytopes
        self.polytopes = []
        for i in range(self.K):
            for j in range(i, self.K):
                # Extract points in cluster
                cluster_inds = (self.labels == i) | (self.labels == j)
                free_cluster = free_pts[cluster_inds]
                if free_cluster.shape[0] < self.min_keep_pts:
                    print("Not enough points to transform.")
                    continue
                # Compute point closest to center of cluster
                b = np.mean(free_cluster, axis=0)
                ind = np.argmin(np.linalg.norm(free_cluster - b, axis=-1))
                b = free_cluster[ind, :]

                # Compute approximate bounding ellipse using 'covariance'
                # A = np.cov(free_pts[cluster_inds], rowvar=False)
                var = free_cluster - b
                A = 1 / (free_cluster.shape[0] - 1) * var.T @ var
                d, v = np.linalg.eig(A)

                # Fit the polytope
                self.fit_polytope(
                    v @ np.diag(np.minimum(np.sqrt(d), self.max_eig)) * np.sqrt(5.991),
                    b
                )
    
    def fit_polytope(self, A: np.ndarray, b: np.ndarray):
        """Fits a convex polytope within the free space using 

        Args:
            A (np.ndarray): A matrix describing the axes of the ellipse
            b (np.ndarray): b vector describing the center of the ellipse
        """
        # Select points to transform
        occ_points = (np.linalg.inv(A) @ ((self.locs + self.map_origin).T - b[:, None])).T
        norm_points = np.linalg.norm(occ_points, axis=-1, keepdims=True)
        keep_pts = (np.squeeze(norm_points) <= self.buffer_mult) & np.logical_not(self.free_mask)
        if np.sum(keep_pts) < self.min_keep_pts:
            print("Not enough points to transform.")
            return
        occ_points = occ_points[keep_pts, :]
        norm_points = norm_points[keep_pts] + 1e-6

        # Transform the points
        trans_points = occ_points * (2 * self.buffer_mult - norm_points) / norm_points
        # Compute the convex hull and obtain star points
        conv_hull = ConvexHull(trans_points)
        star_pts = (self.locs + self.map_origin)[keep_pts][conv_hull.vertices]
        # star_pts_norm = np.linalg.norm(star_pts, axis=-1, keepdims=True)
        # star_pts *= (2 * self.buffer_mult - star_pts_norm) / star_pts_norm

        # star_pts = (A @ star_pts.T + b[:, None]).T

        # Convexify the star-convex set
        A_poly, b_poly = self.compute_polytope_from_points(star_pts)

        # Ensure 'center' point is inside the polytope
        if np.all(A_poly @ b < b_poly):
            # Reduce the polygon to minimal representation
            mat = cdd.Matrix(np.hstack([b_poly[:, None], -A_poly]).tolist(), number_type='float')
            mat.rep_type = cdd.RepType.INEQUALITY
            mat.canonicalize()
            poly = cdd.Polyhedron(mat)

            # Extract inequalities and extreme points
            ext_pts = np.array(poly.get_generators())[:, 1:]
            ineq = np.array(poly.get_inequalities())
            A_poly, b_poly = -ineq[:, 1:], ineq[:, 0]
            norm_A = np.linalg.norm(A_poly, axis=-1, keepdims=True)
            A_poly /= norm_A
            b_poly /= norm_A.squeeze()

            # Reorder extreme points such that ith constraint connects i, i+1 ext pt
            n = ext_pts.shape[0]
            inds = np.array([np.arange(n), (np.arange(n) - 1) % n])        # Handles wrap-around for last element
            sat_mat = A_poly @ ext_pts.T - b_poly[:, None]
            sat_mask = np.all(np.abs(sat_mat[inds, :]) < 1e-10, axis=0)    # Identify which constraints each extreme point is on
            assert np.all(np.sum(sat_mask, axis=1) == 1), "Each row should have exactly one True value"
            ext_pts = np.repeat(ext_pts[None, :, :], n, axis=0)[sat_mask]  # And index them in the proper order
            
            # Each column of `sat_mask` should have exactly one `True`, which selects the correct ext_pts row
            # if not np.all(np.sum(sat_mask, axis=1) == 1):
            #     import scipy.io
            #     print("Each row should have exaclty one true value")
            #     print(star_pts)
            #     conv_hull = ConvexHull(star_pts)
            #     extreme_star_pts = star_pts[conv_hull.vertices]
            #     print(extreme_star_pts)
            #     print(A_poly)
            #     print(b_poly)
            #     print(ext_pts)
            #     scipy.io.savemat('data.mat', {'star_pts' : star_pts, 'extreme_star_pts' : extreme_star_pts, 'A_poly' : A_poly, 'b_poly' : b_poly, 'ext_pts' : ext_pts })
            
            self.polytopes.append({'vertices': ext_pts, 'A': A_poly, 'b': b_poly})  # A <= b

    @staticmethod
    def compute_polytope_from_points(star_pts):
        """Computes a convex polytope inscribed within the star-convex set

        Args:
            star_pts (np.ndarray): extreme points of star-convex set, in clockwise order

        Returns:
            tuple: A, b such that polytobe is Ax <= b
        """
        # TODO: remove redundant constraints, efficient computations?
        # convex hull of star points
        conv_hull = ConvexHull(star_pts)

        # Initialize variables
        vrt = conv_hull.vertices
        constraint_ind = 0
        max_buffer = 0
        star_ind_0 = vrt[-1]
        A = np.zeros((conv_hull.equations.shape[0], 2))
        b = np.zeros((conv_hull.equations.shape[0],))
        a0, b0 = LocalMapper.get_constraint(constraint_ind, star_pts, vrt)

        # Loop over the constraints
        for i in range(star_pts.shape[0]):
            star_ind = (star_ind_0 + i + 1) % star_pts.shape[0]
            # If you have reached the other end of the constraint, buffer by the max buffer
            if star_ind == vrt[constraint_ind]:
                A[constraint_ind] = a0
                b[constraint_ind] = b0 + max_buffer
                max_buffer = 0
                constraint_ind += 1
                a0, b0 = LocalMapper.get_constraint(constraint_ind, star_pts, vrt)
            else:  # Otherwise, increase the maximum buffer if necessary
                buf = np.dot(a0, star_pts[star_ind]) - b0
                if buf < max_buffer:
                    max_buffer = buf

        return A, b

    @staticmethod
    def get_constraint(i:int, star_pts:np.ndarray, vrt:np.ndarray):
        """Get the i'th constraint from the star points

        Args:
            i (int): Constraint to get
            star_pts (np.ndarray): star points
            vrt (np.ndarray): connectivity of star conv hull points

        Returns:
            tuple: a0, b such that a0 <= b
        """
        a0 = star_pts[vrt[(i - 1) % vrt.shape[0]]] - star_pts[vrt[i % vrt.shape[0]]]
        a0 = np.array([-a0[1], a0[0]]) / np.linalg.norm(a0)
        b = np.dot(a0, star_pts[vrt[i % vrt.shape[0]]])
        return a0, b

    @staticmethod 
    def compute_line_intersection(a0, b0, a1, b1):
        """computes the intersection of two lines 

        Args:
            a0 (np.ndarray): first line slope
            b0 (float): first line intercept
            a1 (np.ndarray): second line slope
            b1 (float): second line intercept

        Raises:
            RuntimeError: Parallel lines detected

        Returns:
            np.ndarray: intersection point
        """
        d1 = min(abs(a0[1]), abs(a1[0] - a1[1] * a0[0] / a0[1]))
        d2 = min(abs(a0[0]), abs(a1[1] - a1[0] * a0[1] / a0[0]))
        if d1 > d2:
            if abs(d1 < 1e-6):
                print("Oh no lines parallel, d1")
                print(a0, b0, a1, b1)
                raise RuntimeError("Lines Parallel")
            x = (b1 - a1[1] / a0[1] * b0) / (a1[0] - a1[1] * a0[0] / a0[1])
            y = (b0 - a0[0] * x) / a0[1]
        else:
            if abs(d2 < 1e-6):
                print("Oh no lines parallel, d2")
                print(a0, b0, a1, b1)
                raise RuntimeError("Lines Parallel")
            y = (b1 - a1[0] / a0[0] * b0) / (a1[1] - a1[0] * a0[1] / a0[0])
            x = (b0 - a0[1] * y) / a0[0]
        return np.array([x, y])
    
    def update_occ_grid(self, pos: np.ndarray):
        """_summary_

        Args:
            camera_pos (np.ndarray): Position of the camera in 3D space (for aligning depth map)
            camera_quat (np.ndarray): Orientation of the camera in 3D space (for align)
        """
        # If hopper has moved too far from the center of the occ grid, re-center it
        center_pos = pos[:2] - self.map_center
        if np.linalg.norm(center_pos) > self.recenter_thresh:
            # cells_to_move = center_pos // self.disc
            dx, dy = -int(center_pos[0] // self.disc), -int(center_pos[1] // self.disc)
            print(dx)
            print(type(dx))
            self.map_origin -= np.array([dx * self.disc, dy * self.disc])
            with self.occ_grid_lock:
                if dx > 0:    # Move X up
                    self.occ_grid[:, dx:] = self.occ_grid[:, :-dx]
                    self.occ_grid[:, :dx] = self.UNKNOWN
                else:
                    dx = abs(dx)
                    self.occ_grid[:, :-dx] = self.occ_grid[:, dx:]
                    self.occ_grid[:, -dx:] = self.UNKNOWN

                if dy > 0:    # Move X up
                    self.occ_grid[dy:, :] = self.occ_grid[:-dy, :]
                    self.occ_grid[:dy, :] = self.UNKNOWN
                else:
                    dy = abs(dy)
                    self.occ_grid[:-dy, :] = self.occ_grid[dy:, :]
                    self.occ_grid[-dy:, :] = self.UNKNOWN

        # Mark free points as free, occ_points as occ
        with self.trav_seg_lock:
            free_pts = self.trav_seg.get_free_xy()
            occ_pts = self.trav_seg.get_occ_xy()
        
        free_pts = self.crop_xy_to_map(free_pts)
        occ_pts = self.crop_xy_to_map(occ_pts)

        with self.occ_grid_lock:
            self.occ_grid[self.xy_to_ind_(free_pts)] = self.FREE
            self.occ_grid[self.xy_to_ind_(occ_pts)] = self.OCC

    def crop_xy_to_map(self, pts):
        """Elimiates points outside of the current map boundary

        Args:
            pts (np.ndarray): points in 2D space

        Returns:
            np.ndarray: points which lie inside the boundary of the map
        """
        in_map = (pts[:, 0] >= self.map_origin[0]) & (pts[:, 0] <= self.map_origin[0] + self.disc * self.occ_grid.shape[1]) \
                 & (pts[:, 1] >= self.map_origin[1]) & (pts[:, 1] <= self.map_origin[1] + self.disc * self.occ_grid.shape[0])
        return pts[in_map, :]
    
    def xy_to_ind_(self, pts):
        """Converts x, y points to indices of the corrosponding occupancy grid cells

        Args:
            pts (np.ndarray): points to map to grid indices

        Returns:
            tuple: (y_inds, x_inds) to index occupancy grid
        """
        inds = np.floor((pts - self.map_origin + self.disc / 2) / self.disc).astype(int)
        x_inds = np.clip(inds[:, 0], 0, self.occ_grid.shape[1] - 1)
        y_inds = np.clip(inds[:, 1], 0, self.occ_grid.shape[0] - 1)
        return y_inds, x_inds
    
    def get_xyz(self):
        """Thread-safe get of the trav_seg pointcloud

        Returns:
            np.ndarray: trav_seg point cloud
        """
        with self.trav_seg_lock:
            return self.trav_seg.xyz.copy()
        
    def get_seg_frame(self):
        """Thread-safe get of the trav_seg segment frame

        Returns:
            np.ndarray: trav_seg segment frame
        """
        with self.trav_seg_lock:
            return self.trav_seg.seg_frame.copy()
        
    def get_local_prompt(self):
        """Thread-safe get of the trav_seg local_prompt

        Returns:
            bool: trav_seg local_prompt
        """
        with self.trav_seg_lock:
            return self.trav_seg.local_prompt
        
    def get_prompt_completed(self):
        """Thread-safe get of the trav_seg prompt_completed

        Returns:
            bool: trav_seg prompt_completed
        """
        with self.trav_seg_lock:
            return self.trav_seg.prompt_completed

    def get_all_mask(self):
        """Thread-safe get of the trav_seg all_mask

        Returns:
            np.ndarray: trav_seg all_mask
        """
        with self.trav_seg_lock:
            return self.trav_seg.all_mask.copy()
        
    def add_prompt(self, group:int, label:int, x:int, y:int):
        """Thread-safe call of the trav_seg add_prompt

        Args:
            group (int): object group
            label (int): object label
            x (int): x point position
            y (int): y point position
        """
        with self.trav_seg_lock:
            self.trav_seg.add_prompt(group, label, x, y)

    def shutdown(self):
        self.trav_seg.shutdown()

    def __del__(self):
        """Deletes this object
        """
        self.shutdown()


if __name__ == '__main__':
    star_pts = np.array([
        [-1., -1.],
        [1, -1], 
        [1, 1], 
        [-1, 1],
        [-0.5, 0]
    ])

    A, b = LocalMapper.compute_polytope_from_points(star_pts)
    halfspaces = HalfspaceIntersection(np.hstack((A, -b[:, None])), np.zeros((2,)))
    ext_pts = halfspaces.intersections

     # Reorder extreme points such that ith constraint connects i, i+1 ext pt
    n = ext_pts.shape[0]
    inds = np.array([np.arange(n), (np.arange(n) - 1) % n])  # Handles wrap-around for last element

    # Shape: (2, num_constraints, num_points)
    sat_mat = A @ ext_pts.T - b[:, None]
    sat_mask = np.all(sat_mat[inds, :] == 0, axis=0)  

    # Each column of `sat_mask` should have exactly one `True`, which selects the correct ext_pts row
    assert np.all(np.sum(sat_mask, axis=1) == 1), "Each row should have exactly one True value"

    # Use boolean indexing to retrieve the valid ext_pts
    ext_pts = np.repeat(ext_pts[None, :, :], n, axis=0)[sat_mask]
    
    print(A); print(b); print(ext_pts)
    
    # print(A.shape, b.shape, ext_pts.shape)
    # print(A); print(b); print(ext_pts)
    # print(A @ ext_pts.T - b[:, None])

    # sat_mat = A @ ext_pts.T - b[:, None]
    # ext_pts2 = np.zeros_like(ext_pts)
    # for i in range(ext_pts.shape[0]):
    #     # ext_pts[i] statifies constraints i, i + 1
    #     inds = [i, i - 1] if i - 1 >= 0 else [i, ext_pts.shape[0] - 1]
    #     mask = np.all(sat_mat[inds, :] == 0, axis=0)
    #     assert np.sum(mask) == 1
    #     ext_pts2[i, :] = ext_pts[mask, :]

    # print(ext_pts2)

    # n = ext_pts.shape[0]
    # inds = np.array([np.arange(n), (np.arange(n) - 1) % n])  # Handles wrap-around for last element
    # print("inds", inds)

    # # Shape: (2, num_constraints, num_points)
    # sat_mask = np.all(sat_mat[inds, :] == 0, axis=0)  
    # print('satmat', sat_mat)
    # print('sat_mask', sat_mask)

    # # Each column of `sat_mask` should have exactly one `True`, which selects the correct ext_pts row
    # assert np.all(np.sum(sat_mask, axis=1) == 1), "Each row should have exactly one True value"

    # # Use boolean indexing to retrieve the valid ext_pts
    # ext_pts2 = np.repeat(ext_pts[None, :, :], n, axis=0)[sat_mask]

    # print(ext_pts2)
    # print('here')
