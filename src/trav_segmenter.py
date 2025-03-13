import cv2
import time
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor


class TravSegmenter:

    def __init__(
            self, width=640, height=480, frame_rate=30,
            rnsc_dist_thres=0.05, rnsc_n0=3, rnsc_iters=1000,
            record=False, record_fn='output.bag', print_timing=False,
            from_file=False, input_file=None, loop_playback=False,
            o3d_vis=False,
            checkpoint="/home/noelcs/repos/my_tam/checkpoints/efficienttam_ti_512x512.pt",
            model_cfg="configs/efficienttam/efficienttam_ti_512x512.yaml"):
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
        self.depth_frame = None
        self.color_frame = None
        self.pc = rs.pointcloud()
        self.pcd = o3d.geometry.PointCloud()
        self.plane_model = None
        self.inliers = None
        self.print_timing = print_timing

        # RANSAC parameters
        self.rnsc_dist_thresh = rnsc_dist_thres
        self.rnsc_n0 = rnsc_n0
        self.rnsc_iters = rnsc_iters

        # Align color and depth images
        self.align = rs.align(rs.stream.color)

        # O3D Visualization
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
        self.selected_point = None
        self.click_points, self.click_labels = {}, {}

        # CV2 window for camera control
        self.curr_group = 0
        
        self.seg_vis_win_name = "Camera Control"
        cv2.namedWindow(self.seg_vis_win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.seg_vis_win_name, self.on_mouse_)
    
    def start_pipeline(self):
        if not self.pipeline_started:
            self.pipeline.start(self.config)
            self.pipeline_started = True

    def capture_frame(self):
        t0 = time.perf_counter_ns()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        if self.print_timing:
            print(f"Timing Image Cap: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def fit_ground_plane(self): 
        if self.depth_frame is None:
            raise RuntimeError("No images have been captured. Cannot identify ground plane.")
        t0 = time.perf_counter_ns()
        # self.pc.map_to(self.color_frame)  # Map to color for RGB information
        points = self.pc.calculate(self.depth_frame)

        # Convert to numpy array
        v = np.asanyarray(points.get_vertices())  # xyz points
        xyz = np.column_stack((v['f0'], v['f1'], v['f2']))
        xyz = xyz.astype(np.float64, copy=False)  # Ensure correct dtype without unnecessary copy, critical for timing
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=self.rnsc_dist_thresh,
                                                      ransac_n=self.rnsc_n0,
                                                      num_iterations=self.rnsc_iters)
        if self.print_timing:
            print(f"Timing RANSAC Fit: {(time.perf_counter_ns() - t0) / 1e6}ms")
        
    def o3d_vis_ground_plane(self):
        # Step 4: Extract inliers (ground) and outliers (obstacles)
        ground_cloud = self.pcd.select_by_index(self.inliers)
        obstacle_cloud = self.pcd.select_by_index(self.inliers, invert=True)

        # Step 5: Visualize
        o3d.visualization.draw_geometries([ground_cloud.paint_uniform_color([0,1,0]), 
                                        obstacle_cloud.paint_uniform_color([1,0,0])])
        
    def vis_frame(self):
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

    def select_pixel_(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            self.selected_point = np.array([[x,y]], np.int)
            print(f"Selected point: ({x}, {y})")

    def prompt_seg(self):
        color_image = np.asanyarray(self.color_frame.get_data())
        self.seg_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.predictor.load_first_frame(self.seg_frame)
        while self.curr_group != 13:  # for 'enter' key
            self.update_seg_vis()

    def segment_frame(self):
        t0 = time.perf_counter_ns()
        color_image = np.asanyarray(self.color_frame.get_data())
        self.seg_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.out_obj_ids, self.out_mask_logits = self.predictor.track(self.seg_frame)
        if self.print_timing:
            print(f"Segmenting: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def update_seg_vis(self):
        t0 = time.perf_counter_ns()
        if self.out_obj_ids is None:
            frame = self.seg_frame
        else:
            width, height = self.seg_frame.shape[:2][::-1]
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(self.out_obj_ids)):
                out_mask = (self.out_mask_logits[i]>0.0).permute(1,2,0).cpu().numpy().astype(np.uint8)*255
                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(self.seg_frame, 1, all_mask, 0.5, 0)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow(self.seg_vis_win_name, frame)
        self.curr_group = cv2.waitKey(1)
        if self.print_timing:
            print(f"Updating Seg Vis: {(time.perf_counter_ns() - t0) / 1e6}ms")

    def on_mouse_(self, event, x, y, flags, param):
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
        print('here')

    def shutdown(self):
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
        cv2.destroyAllWindows()
        self.vis.destroy_window()
        
    def __del__(self):
        self.shutdown()
