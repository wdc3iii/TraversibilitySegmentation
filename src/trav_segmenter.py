import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class TravSegmenter:

    def __init__(self, width=640, height=480, frame_rate=30, rnsc_dist_thres=0.05, rnsc_n0=3, rnsc_iters=1000):
        # Set up the pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, frame_rate)
        self.pipeline_started = False
        self.start_pipeline()

        # Frames and pointclouds
        self.depth_frame = None
        self.color_frame = None
        self.pc = rs.pointcloud()
        self.pcd = o3d.geometry.PointCloud()
        self.plane_model = None
        self.inliers = None

        # RANSAC parameters
        self.rnsc_dist_thresh = rnsc_dist_thres
        self.rnsc_n0 = rnsc_n0
        self.rnsc_iters = rnsc_iters

        # Align color and depth images
        self.align = rs.align(rs.stream.color)
    
    def start_pipeline(self):
        if not self.pipeline_started:
            self.pipeline.start(self.config)
            self.pipeline_started = True

    def capture_image(self):
        t0 = time.perf_counter_ns()
        frames = self.pipeline.wait_for_frames()
        t1 = time.perf_counter_ns()
        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()
        t2 = time.perf_counter_ns()

        # self.pc.map_to(self.color_frame)  # Map to color for RGB information
        points = self.pc.calculate(self.depth_frame)
        t3 = time.perf_counter_ns()

        # Convert to numpy array
        v = np.asanyarray(points.get_vertices())  # xyz points
        xyz = np.array([(p[0], p[1], p[2]) for p in v])  # Convert to structured np.array
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        t4 = time.perf_counter_ns()
        print(f"Timing Image Cap: {(t4 - t0) / 1e6}ms\n\tWait Frames: {(t1 - t0) / 1e6}\n\tExtr Frames: {(t2 - t1) / 1e6}\n\tComp Points {(t3 - t2) / 1e6}\n\tTrans Points: {(t4 - t3) / 1e6}")

    def fit_ground_plane(self): 
        if self.depth_frame is None:
            raise RuntimeError("No images have been captured. Cannot identify ground plane.")
        t0 = time.perf_counter_ns()
        self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=self.rnsc_dist_thresh,
                                                      ransac_n=self.rnsc_n0,
                                                      num_iterations=self.rnsc_iters)
        print(f"Timing RANSAC Fit: {(time.perf_counter_ns() - t0) / 1e6}ms")
        
    def o3d_vis_ground_plane(self):
        # Step 4: Extract inliers (ground) and outliers (obstacles)
        ground_cloud = self.pcd.select_by_index(self.inliers)
        obstacle_cloud = self.pcd.select_by_index(self.inliers, invert=True)

        # Step 5: Visualize
        o3d.visualization.draw_geometries([ground_cloud.paint_uniform_color([0,1,0]), 
                                        obstacle_cloud.paint_uniform_color([1,0,0])])
        
    def shutdown(self):
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
        
    def __del__(self):
        self.shutdown()
