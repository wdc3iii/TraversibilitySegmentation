from datetime import datetime
from trav_seg.trav_segmenter import TravSegmenter


timestamp = datetime.now().strftime("%m-%d_%H-%M")
fn = f"output/depth_data_{timestamp}.mat"
# Initialize segmenter
trav_seg = TravSegmenter()

# Prompt segmenter
trav_seg.capture_frame()
trav_seg.prompt_seg()

# Segment and save
trav_seg.segment_frame()
print(f"Saving frame to: {fn}")
trav_seg.save_frame()

# Stop pipeline
trav_seg.shutdown()
