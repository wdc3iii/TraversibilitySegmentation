import cv2
import time
import torch
import scipy.io
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor


class TravSegmenter:

    def __init__(
            self, width=640, height=480, frame_rate=30, min_depth=0.1,
            rnsc_dist_thres=0.05, rnsc_n0=3, rnsc_iters=1000,
            record=False, record_fn='output.bag', print_timing=False,
            from_file=False, input_file=None, loop_playback=False,
            o3d_vis=False, local_prompt=False,
            checkpoint="/home/noelcs/repos/my_tam/checkpoints/efficienttam_ti_512x512.pt",
            model_cfg="configs/efficienttam/efficienttam_ti_512x512.yaml"):
        """Instantiates an object to segment traversible regions from the environment

        Args:
            width (int, optional): _description_. Defaults to 640.
            height (int, optional): _description_. Defaults to 480.
            frame_rate (int, optional): _description_. Defaults to 30.
            min_depth (float, optional): _description_. Defaults to 0.1.
            rnsc_dist_thres (float, optional): _description_. Defaults to 0.05.
            rnsc_n0 (int, optional): _description_. Defaults to 3.
            rnsc_iters (int, optional): _description_. Defaults to 1000.
            record (bool, optional): _description_. Defaults to False.
            record_fn (str, optional): _description_. Defaults to 'output.bag'.
            print_timing (bool, optional): _description_. Defaults to False.
            from_file (bool, optional): _description_. Defaults to False.
            input_file (_type_, optional): _description_. Defaults to None.
            loop_playback (bool, optional): _description_. Defaults to False.
            o3d_vis (bool, optional): _description_. Defaults to False.
            local_prompt(bool, optional): _description_. Defaults to False.
            checkpoint (str, optional): _description_. Defaults to "/home/noelcs/repos/my_tam/checkpoints/efficienttam_ti_512x512.pt".
            model_cfg (str, optional): _description_. Defaults to "configs/efficienttam/efficienttam_ti_512x512.yaml".
        """
        # Set up the pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if from_file:
            assert input_file is not None
            self.config.enable_device_from_file(input_file, repeat_playback=loop_playback)
        else:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, frame_rate)
        if record:
            self.config.enable_record_to_file(record_fn)

        self.pipeline_started = False
        self.start_pipeline()

        # Frames and pointclouds
        self.min_depth = min_depth
        self.depth_frame = None
        self.color_frame = None
        self.pc = rs.pointcloud()
        self.pcd = o3d.geometry.PointCloud()
        self.xyz = None
        self.plane_model = None
        self.inliers = None
        self.print_timing = print_timing
        self.save_flag = False

        # RANSAC parameters
        self.rnsc_dist_thresh = rnsc_dist_thres
        self.rnsc_n0 = rnsc_n0
        self.rnsc_iters = rnsc_iters

        # Align color and depth images
        self.align = rs.align(rs.stream.color)

        # O3D Visualization
        self.o3d_vis = o3d_vis
        if o3d_vis:
            self.ground_cloud = o3d.geometry.PointCloud()
            self.obstacle_cloud = o3d.geometry.PointCloud()
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.added_geoms = False
        
        # TAM 
        self.predictor = build_efficienttam_camera_predictor(
            model_cfg, 
            checkpoint, 
            device=torch.device("cuda"),
            vos_optimized=True,
            hydra_overrides_extra=["++model.compile_image_encoder=False"], 
            apply_postprocessing=False,
        )
        self.out_obj_ids, self.out_mask_logits, self.seg_frame = None, None, None
        self.click_points, self.click_labels = {}, {}

        # CV2 window for camera control
        self.curr_group = 0
        self.reprompt_segment = False
        self.local_prompt = local_prompt
        if self.local_prompt:
            self.seg_vis_win_name = "Camera Control"
            cv2.namedWindow(self.seg_vis_win_name, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.seg_vis_win_name, (960, 540))
            cv2.setMouseCallback(self.seg_vis_win_name, self.on_mouse_)
    
    def start_pipeline(self):
        """Starts the camera pipeline (if not already started)
        """
        if not self.pipeline_started:
            self.pipeline.start(self.config)
            self.pipeline_started = True

    def capture_frame(self):
        """Captures the most recent frame from the camera pipeline, and extracts point cloud information
        """
        t0 = time.perf_counter_ns()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()
        
        points = self.pc.calculate(self.depth_frame)

        # Convert to numpy array
        v = np.asanyarray(points.get_vertices())  # xyz points
        self.xyz = np.column_stack((v['f0'], v['f1'], v['f2'])).astype(np.float64, copy=False)  # Ensure correct dtype without unnecessary copy, critical for timing
        self.xyz = self.xyz[np.linalg.norm(self.xyz, axis=-1, keepdims=True) >= self.min_depth]
        if self.reprompt_segment:
            self.prompt_seg()
            self.reprompt_segment = False

        if self.print_timing:
            print(f"Timing Image Cap: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def prompt_seg(self):
        """Allows the user to select a set of points with which to prompt the segmenter.
        """
        if not self.local_prompt:
            raise RuntimeError("Trav Segmenter not initialized to allow local prompting. Prompt via add_prompt()")
        if self.color_frame is None:
            self.capture_frame()
        color_image = np.asanyarray(self.color_frame.get_data())
        self.seg_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.predictor.load_first_frame(self.seg_frame)
        while self.curr_group != 13:  # for 'enter' key
            self.generate_segment_mask_()
            self.update_seg_vis()

    def segment_frame(self):
        """Segments the current frame
        """
        t0 = time.perf_counter_ns()
        color_image = np.asanyarray(self.color_frame.get_data())
        self.seg_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.out_obj_ids, self.out_mask_logits = self.predictor.track(self.seg_frame)
        self.generate_segment_mask_()

        if self.print_timing:
            print(f"Segmenting: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def save_frame(self, save_path):
        """Saves the current frame (color, point cloud, mask)"""
        scipy.io.savemat(save_path, {"rgb": self.color_frame, "pc": self.xyz, "mask": self.all_mask})

    def get_free_xy(self):
        return self.xyz[self.flat_mask, :2]

    def get_occ_xy(self):
        return self.xyz[np.logical_not(self.flat_mask), :2]

    def transform_point_cloud_(self, p: np.ndarray, R: np.ndarray):
        """Transforms the current point cloud by the given transform

        Args:
            p (np.ndarray): (3,) position array for the transformation
            R (np.ndarray): (3, 3) rotation matrix for the transformation
        """
        self.xyz = (R @ self.xyz.T).T + p
        
    def generate_segment_mask_(self):
        """Generates the segmentation mask from the output of the segmenter.
        """
        if self.out_obj_ids is None:
            return

        self.all_mask = np.any((self.out_mask_logits.permute(0, 2, 3, 1) > 0.0).cpu().numpy(), axis=0).astype(bool)
        self.flat_mask = self.all_mask.reshape((-1, ), order='C')
    
    def add_prompt(self, group, label, x, y):
        event = cv2.EVENT_LBUTTONDOWN if label else cv2.EVENT_RBUTTONDOWN
        self.curr_group = group
        self.on_mouse_(event, x, y, None, None)

    def on_mouse_(self, event, x, y, flags, param):
        """Mouse callback

        Args:
            event (_type_): mouse event
            x (_type_): x location associated with event
            y (_type_): y location associated with event
            flags (_type_): flag associated with event
            param (_type_): parameters associated with the event
        """
        if event == cv2.EVENT_MBUTTONDOWN:
            self.reprompt_segment = True
            return
        if event != cv2.EVENT_LBUTTONDOWN and event != cv2.EVENT_RBUTTONDOWN:
            return
        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        
        new_point = np.array([[x, y]], dtype=np.int32)
        new_label = np.array([label], dtype=np.int32) 
        if self.curr_group not in self.click_points.keys():
            self.click_points[self.curr_group] = np.empty((0, 2), dtype=np.int32)
            self.click_labels[self.curr_group] = np.empty(0, dtype=np.int32)
        self.click_points[self.curr_group] = np.append(self.click_points[self.curr_group], new_point, axis=0)
        self.click_labels[self.curr_group] = np.append(self.click_labels[self.curr_group], new_label)
        _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_prompt(frame_idx=0, obj_id=self.curr_group, points=new_point,labels=new_label)

    def shutdown(self):
        """Shuts down the camera pipeline
        """
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
        cv2.destroyAllWindows()
        if self.o3d_vis:
            self.vis.destroy_window()
        
    def __del__(self):
        """Deletes this object
        """
        self.shutdown()

    def update_seg_vis(self):
        """Updates the segmentation visualizer.
        """
        t0 = time.perf_counter_ns()
        if self.out_obj_ids is None:
            frame = self.seg_frame
        else:
            frame = cv2.addWeighted(
                self.seg_frame,
                1,
                cv2.cvtColor(self.all_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB),
                0.5,
                0
            )

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow(self.seg_vis_win_name, frame)
        self.curr_group = cv2.waitKey(1)

        if self.print_timing:
            print(f"Updating Seg Vis: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def vis_frame(self):
        """Visualize the current frame
        """
        depth_image = np.asanyarray(self.depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
    
    def update_pc_vis(self):
        """Visualize the current pointcloud
        """
        t0 = time.perf_counter_ns()
        
        self.ground_cloud.points = self.pcd.select_by_index(self.inliers).points
        self.ground_cloud.paint_uniform_color([0, 1, 0])
        
        self.obstacle_cloud.points = self.pcd.select_by_index(self.inliers, invert=True).points
        self.obstacle_cloud.paint_uniform_color([1, 0, 0])

        if not self.added_geoms:
            self.vis.add_geometry(self.ground_cloud)
            self.vis.add_geometry(self.obstacle_cloud)
            self.added_geoms = True

        self.vis.update_geometry(self.ground_cloud)
        self.vis.update_geometry(self.obstacle_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

        if self.print_timing:
            print(f"Timing Update Vis: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def fit_ground_plane(self): 
        """Fit a ground plane to the point cloud

        Raises:
            RuntimeError: Trying to fit a groud plane before an image has been captured.
        """
        if self.depth_frame is None:
            self.capture_frame()
        t0 = time.perf_counter_ns()
        # self.pc.map_to(self.color_frame)  # Map to color for RGB information
        self.pcd.points = o3d.utility.Vector3dVector(self.xyz)
        self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=self.rnsc_dist_thresh,
                                                      ransac_n=self.rnsc_n0,
                                                      num_iterations=self.rnsc_iters)
        if self.print_timing:
            print(f"Timing RANSAC Fit: {(time.perf_counter_ns() - t0) / 1e6}ms")
        
    def o3d_vis_ground_plane(self):
        """Visualize the current ground plane estimate.
        """
        # Step 4: Extract inliers (ground) and outliers (obstacles)
        ground_cloud = self.pcd.select_by_index(self.inliers)
        obstacle_cloud = self.pcd.select_by_index(self.inliers, invert=True)

        # Step 5: Visualize
        o3d.visualization.draw_geometries([ground_cloud.paint_uniform_color([0,1,0]), 
                                        obstacle_cloud.paint_uniform_color([1,0,0])])
