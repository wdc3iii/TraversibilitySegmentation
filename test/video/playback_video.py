from src.trav_segmenter import TravSegmenter

input_file = "output/d435_03-13_10-28.bag"
trav_seg = TravSegmenter(from_file=True, input_file=input_file)

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        trav_seg.capture_frame()
        trav_seg.vis_frame()
except KeyboardInterrupt:
    print("\nRecording stopped.")

# Stop pipeline
trav_seg.shutdown()