from datetime import datetime
from trav_seg.trav_segmenter import TravSegmenter

timestamp = datetime.now().strftime("%m-%d_%H-%M")
trav_seg = TravSegmenter(record=True, record_fn="output/d435_" + timestamp + ".bag")

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        trav_seg.capture_frame()
        trav_seg.vis_frame()
except KeyboardInterrupt:
    print("\nRecording stopped.")

# Stop pipeline
trav_seg.shutdown()