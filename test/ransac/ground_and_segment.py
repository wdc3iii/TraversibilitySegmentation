from datetime import datetime
from trav_seg.trav_segmenter import TravSegmenter

od3_vis = True
print_timing = True

# From Camera
record = False
timestamp = datetime.now().strftime("%m-%d_%H-%M")
trav_seg = TravSegmenter(record=record, record_fn="output/d435_" + timestamp + ".bag", o3d_vis=od3_vis, print_timing=print_timing)

# From Recording
# input_file = "output/d435_03-13_11-03.bag"
# trav_seg = TravSegmenter(from_file=True, input_file=input_file, o3d_vis=od3_vis, print_timing=print_timing)

trav_seg.capture_frame()
trav_seg.prompt_seg()
try:
    while True:
        # First, capture the frame
        trav_seg.capture_frame()

        # Run RANSAC
        trav_seg.fit_ground_plane()

        # Run Segmenting
        trav_seg.segment_frame()

        # Update Vis
        if od3_vis:
            trav_seg.update_pc_vis()
        trav_seg.update_seg_vis()
        print("\n")

except KeyboardInterrupt:
    print("\nRecording stopped.")

# Stop pipeline
trav_seg.shutdown()