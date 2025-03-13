from datetime import datetime
from src.trav_segmenter import TravSegmenter

# From Camera
# record = True
# timestamp = datetime.now().strftime("%m-%d_%H-%M")
# trav_seg = TravSegmenter(record=record, record_fn="output/d435_" + timestamp + ".bag", o3d_vis=True, print_timing=True)

# From Recording
input_file = "output/d435_03-13_11-03.bag"
trav_seg = TravSegmenter(from_file=True, input_file=input_file, o3d_vis=True, print_timing=True)

try:
    while True:
        trav_seg.capture_frame()

        trav_seg.fit_ground_plane()

        trav_seg.update_pc_vis()
        trav_seg.vis.poll_events()
        trav_seg.vis.update_renderer()
except KeyboardInterrupt:
    print("\nRecording stopped.")

# Stop pipeline
trav_seg.shutdown()