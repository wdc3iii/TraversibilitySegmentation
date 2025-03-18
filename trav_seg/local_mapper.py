import random
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from trav_seg.trav_segmenter import TravSegmenter


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
        self.occ_grid = np.ones((dim, dim), dtype=int) * self.UNKNOWN
        x_locs = np.repeat(np.arange(self.occ_grid.shape[1])[None, :], self.occ_grid.shape[0], axis=0) * disc + disc / 2
        y_locs = np.repeat(np.arange(self.occ_grid.shape[0])[:, None], self.occ_grid.shape[1], axis=1) * disc + disc / 2
        self.locs = np.hstack((x_locs.flatten()[:, None], y_locs.flatten()[:, None]))
        self.disc = disc
        self.recenter_thresh = recenter_thresh
        self.buffer_mult = buffer_mult
        self.max_eig = max_eig
        self.min_keep_pts = min_keep_pts
        self.local_prompt = local_prompt
        self.trav_seg = TravSegmenter(record=False, o3d_vis=False, print_timing=False, local_prompt=self.local_prompt)

        self.K = K
        self.rng = random.Random(42)  # Seed for reproducibility

        mid = dim // 2
        dx = int(initial_free_radius // disc)
        self.occ_grid[mid-dx:mid+dx, mid-dx:mid+dx] = self.FREE

        # Assumes hopper gets dropped at (or around) [0, 0]
        self.map_origin = np.array([-mid *  self.disc, -mid * self.disc])
        
        self.kmeans = None
        self.labels = None
        self.free_mask = None
        if self.local_prompt:
            self.prompt_seg()

    @property
    def map_center(self):
        return self.map_origin + np.array([self.occ_grid.shape[1], self.occ_grid.shape[0]]) // 2 * self.disc
    
    def prompt_seg(self):
        self.trav_seg.capture_frame()
        self.trav_seg.prompt_seg()

    def capture_frame(self, p, R):
        self.trav_seg.capture_frame()
        self.trav_seg.transform_point_cloud_(p, R)

    def segment_frame(self):
        self.trav_seg.segment_frame()

    def fit_free_space(self):
        """Fits a set of K convex polytopes to the free space
        """
        # Separate free and occupied points
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
    
        self.polytopes = []
        for i in range(self.K):
            for j in range(i, self.K):
                # print(f"i={i}, j={j}")
                cluster_inds = (self.labels == i) | (self.labels == j)
                b = np.mean(free_pts[cluster_inds], axis=0)
                A = np.cov(free_pts[cluster_inds], rowvar=False)
                d, v = np.linalg.eig(A)
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
        # TODO: wrong points to transform!!
        trans_points = (np.linalg.inv(A) @ ((self.locs + self.map_origin).T - b[:, None])).T
        norm_points = np.linalg.norm(trans_points, axis=-1, keepdims=True)
        keep_pts = (np.squeeze(norm_points) <= self.buffer_mult) & np.logical_not(self.free_mask)
        if np.sum(keep_pts) < self.min_keep_pts:
            print("Not enough points to transform.")
            return
        trans_points = trans_points[keep_pts, :]
        norm_points = norm_points[keep_pts] + 1e-6

        # Transform the points
        trans_points *= (2 * self.buffer_mult - norm_points) / norm_points
        conv_hull = ConvexHull(trans_points)
        ext_pts = trans_points[conv_hull.vertices]
        ext_pts_norm = np.linalg.norm(ext_pts, axis=-1, keepdims=True)
        ext_pts *= (2 * self.buffer_mult - ext_pts_norm) / ext_pts_norm

        ext_pts = (A @ ext_pts.T + b[:, None]).T

        edges = trans_points[conv_hull.simplices]
        edge_0, edge_1 = edges[:, 0, :], edges[:, 1, :]
        edge = edge_1 - edge_0
        normal = np.hstack([-edge[:, 0, None], edge[:, 1, None]])
        normal = np.where(np.sum(normal * (b[None, :] - edge_0), axis=1, keepdims=True) > 0, -normal, normal)
        offset = np.sum(normal * edge_0, axis=1)
        self.polytopes.append({'vertices': ext_pts, 'A': normal, 'b': offset})  # A <= b

    def update_occ_grid(self, pos: np.ndarray):
        """_summary_

        Args:
            camera_pos (np.ndarray): Position of the camera in 3D space (for aligning depth map)
            camera_quat (np.ndarray): Orientation of the camera in 3D space (for align)
        """
        # If hopper has moved too far from the center of the occ grid, re-center it
        center_pos = pos[:2] - self.map_center
        if np.linalg.norm(center_pos) > self.recenter_thresh:
            self.get_logger().info("Recentering Local Map")
            self.map_origin += center_pos
            cells_to_move = center_pos // self.disc
            dx, dy = cells_to_move[0], cells_to_move[1]
            if dx > 0:    # Move X up
                self.occ_grid[:, dx:] = self.occ_grid[:, :-dx]
                self.occ_grid[:, :dx] = self.UNKNOWN
            else:
                self.occ_grid[:, :-dx] = self.occ_grid[:, dx:]
                self.occ_grid[:, -dx:] = self.UNKNOWN

            if dy > 0:    # Move X up
                self.occ_grid[dy:, :] = self.occ_grid[:-dy, :]
                self.occ_grid[:dy, :] = self.UNKNOWN
            else:
                self.occ_grid[:-dy, :] = self.occ_grid[dy:, :]
                self.occ_grid[-dy:, :] = self.UNKNOWN

        # Mark free points as free
        free_pts = self.trav_seg.get_free_xy()
        free_pts = self.crop_xy_to_map(free_pts)
        self.occ_grid[self.xy_to_ind_(free_pts)] = self.FREE

        # Mark occupied points as occupied
        occ_pts = self.trav_seg.get_occ_xy()
        occ_pts = self.crop_xy_to_map(occ_pts)
        self.occ_grid[self.xy_to_ind_(occ_pts)] = self.OCC

    def crop_xy_to_map(self, pts):
        in_map = (pts[:, 0] >= self.map_origin[0]) & (pts[:, 0] <= self.map_origin[0] + self.disc * self.occ_grid.shape[1]) \
                 & (pts[:, 1] >= self.map_origin[1]) & (pts[:, 1] <= self.map_origin[1] + self.disc * self.occ_grid.shape[0])
        return pts[in_map, :]
    
    def xy_to_ind_(self, pts):
        inds = np.floor((pts - self.map_origin + self.disc / 2) / self.disc).astype(int)
        x_inds = np.clip(inds[:, 0], 0, self.occ_grid.shape[1] - 1)
        y_inds = np.clip(inds[:, 1], 0, self.occ_grid.shape[0] - 1)
        return y_inds, x_inds

    def shutdown(self):
        self.trav_seg.shutdown()

    def __del__(self):
        """Deletes this object
        """
        self.shutdown()
