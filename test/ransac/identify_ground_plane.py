import time
from trav_seg.trav_segmenter import TravSegmenter


N = 10

trav_seg = TravSegmenter()
trav_seg.capture_frame()

t0 = time.perf_counter_ns()
for _ in range(N):
    trav_seg.capture_frame()
    trav_seg.fit_ground_plane()
print(f"Avg Duration: {(time.perf_counter_ns() - t0) / 1e6 / N} ms")

trav_seg.o3d_vis_ground_plane()
trav_seg.shutdown()

# export PYTHONPATH=$PYTHONPATH:$(pwd)