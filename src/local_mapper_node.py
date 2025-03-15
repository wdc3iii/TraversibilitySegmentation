import rclpy
import cv2
import torch
import numpy as np
import asyncio
import threading
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("segmentation_node")

        # TF2 Transform buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for capturing camera frames and transforms
        self.timer = self.create_timer(0.1, self.capture_frame)  # 10Hz

        # Image bridge
        self.bridge = CvBridge()

        # Shared Data and Locks
        self.latest_frame = None
        self.occupancy_grid = np.zeros((100, 100), dtype=np.int8)
        self.lock = threading.Lock()

        # Segmentation Model (EfficientTAM)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmenter = build_efficienttam_camera_predictor(
            model_cfg="configs/efficienttam/efficienttam_ti_512x512.yaml",
            checkpoint="/home/noelcs/repos/my_tam/checkpoints/efficienttam_ti_512x512.pt",
            device=self.device,
            vos_optimized=True,
            hydra_overrides_extra=["++model.compile_image_encoder=False"], 
            apply_postprocessing=False
        )

        # Flags to trigger tasks asynchronously
        self.segmentation_ready = threading.Event()
        self.occupancy_ready = threading.Event()
        self.polytope_ready = threading.Event()

        # Start Threads for Async Execution
        self.segmentation_thread = threading.Thread(target=self.run_segmentation, daemon=True)
        self.occupancy_thread = threading.Thread(target=self.update_occupancy_grid, daemon=True)
        self.polytope_thread = threading.Thread(target=self.compute_free_space_polytopes, daemon=True)

        self.segmentation_thread.start()
        self.occupancy_thread.start()
        self.polytope_thread.start()

    def capture_frame(self):
        """Captures a frame and retrieves transform."""
        with self.lock:
            # Simulated camera frame (Replace with actual ROS 2 camera topic subscription)
            self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Get transform (Replace with real frame IDs)
            try:
                transform = self.tf_buffer.lookup_transform("map", "camera_frame", rclpy.time.Time())
                self.get_logger().info("Captured frame and transform.")
            except Exception as e:
                self.get_logger().warn(f"Could not get transform: {e}")
                return

        # Trigger segmentation
        self.segmentation_ready.set()

    def run_segmentation(self):
        """Runs segmentation asynchronously when triggered."""
        while rclpy.ok():
            self.segmentation_ready.wait()
            self.segmentation_ready.clear()

            with self.lock:
                if self.latest_frame is None:
                    continue
                frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)

            # Run segmentation on GPU
            self.segmenter.load_first_frame(frame_rgb)
            obj_ids, mask_logits = self.segmenter.track(frame_rgb)

            # Convert mask logits to binary segmentation mask
            segmentation_mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255

            self.get_logger().info("Segmentation completed.")

            # Notify occupancy grid update
            self.occupancy_ready.set()

    def update_occupancy_grid(self):
        """Updates occupancy grid asynchronously."""
        while rclpy.ok():
            self.occupancy_ready.wait()
            self.occupancy_ready.clear()

            with self.lock:
                # Simulated occupancy grid update (Replace with real processing)
                self.occupancy_grid = np.random.randint(0, 2, (100, 100), dtype=np.int8)

            self.get_logger().info("Updated occupancy grid.")

            # Notify free-space polytope computation
            self.polytope_ready.set()

    def compute_free_space_polytopes(self):
        """Computes free-space polytopes asynchronously."""
        while rclpy.ok():
            self.polytope_ready.wait()
            self.polytope_ready.clear()

            # Simulated polytope computation (Replace with real convex decomposition)
            self.get_logger().info("Computed free-space polytopes.")

    def shutdown(self):
        """Graceful shutdown."""
        self.segmentation_thread.join()
        self.occupancy_thread.join()
        self.polytope_thread.join()


def main():
    rclpy.init()
    node = SegmentationNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # Multi-threaded event loop
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
