from src.trav_segmenter import TravSegmenter

trav_seg = TravSegmenter()

trav_seg.capture_image()
trav_seg.fit_ground_plane()
trav_seg.o3d_vis_ground_plane()
trav_seg.shutdown()

# export PYTHONPATH=$PYTHONPATH:$(pwd)