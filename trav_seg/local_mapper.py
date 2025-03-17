import numpy as np
from trav_seg.trav_segmenter import TravSegmenter


class LocalMapper:

    FREE = 0
    OCC = 100
    UNKNOWN = -1

    def __init__(self, disc: float, dim: int, K: int, initial_free_radius: float, recenter_thresh):
        """Initializes a LocalMapper object

        Args:
            disc (float): spatial discretization in meters
            dim (int): number of grid cells in the map
            K (int): number of free space regions to identify
            init_free_radius (float): radius around the robot to initially assume is free
        """
        self.occ_grid = np.ones((dim, dim), dtype=int) * self.UNKNOWN
        self.disc = disc
        self.recenter_thresh = recenter_thresh
        self.trav_seg = TravSegmenter(record=False, o3d_vis=False, print_timing=False)

        self.K = K

        mid = dim // 2
        dx = int(initial_free_radius // disc)
        self.occ_grid[mid-dx:mid+dx, mid-dx:mid+dx] = self.FREE

        # Assumes hopper gets dropped at (or around) [0, 0]
        self.map_origin = np.array([-mid *  self.disc, -mid * self.disc])
        
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
        # TODO: Fit the free space
    
    def cluster_free_space_(self):
        """Clusters the free space into K sections using gaussian mixture models
        """
        raise NotImplementedError
    
    def fit_polytope(self, A: np.ndarray, b: np.ndarray):
        """Fits a convex polytope within the free space using 

        Args:
            A (np.ndarray): A matrix describing the axes of the ellipse
            b (np.ndarray): b vector describing the center of the ellipse
        """
        raise NotImplementedError

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
