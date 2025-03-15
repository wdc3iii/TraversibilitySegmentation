import numpy as np
from trav_seg.trav_segmenter import TravSegmenter


class LocalMapper:

    def __init__(self, disc: float, dim: int, K: int, initial_free_radius):
        """Initializes a LocalMapper object

        Args:
            disc (float): spatial discretization in meters
            dim (int): number of grid cells in the map
            K (int): number of free space regions to identify
        """
        self.occ_grid = np.ones((dim, dim))
        self.disc = disc
        self.trav_seg = TravSegmenter(record=False, o3d_vis=False, print_timing=False)

        self.K = K

        mid = dim // 2
        dx = initial_free_radius // disc
        self.occ_grid[mid-dx:mid+dx, mid-dx:mid+dx] = 0

        # Assumes hopper gets dropped at (or around) [0, 0]
        self.map_origin = np.array([-mid *  self.disc, -mid * self.disc])
        
        self.prompt_seg()

    def prompt_seg(self):
        self.trav_seg.capture_frame()
        self.trav_seg.prompt_seg()

    def capture_frame(self, transform):
        self.trav_seg.capture_frame()
        self.trav_seg.transform_point_cloud(transform)

    def segment_frame(self):
        self.trav_seg.segment_frame()

    def fit_free_space(self):
        """Fits a set of K convex polytopes to the free space
        """
        raise NotImplementedError
    
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
        # TODO

        # Mark free points as free
        # TODO

        # Mark occupied points as occupied
        # TODO

    