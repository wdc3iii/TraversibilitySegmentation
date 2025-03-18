import cv2
import numpy as np
from datetime import datetime
from trav_seg.local_mapper import LocalMapper

window_name = "Occupancy Grid with Polytopes"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 1600)  # Set window size
im_scale = 3
local_mapper = LocalMapper(0.05, 200, 5, 0.25, 0.5, local_prompt=True)

try:
    while True:
        # First, capture the frame
        print("Capture Frame")
        R = np.array([
            [-0.38651647, -0.51541878, -0.76481926],
            [ 0.23052047,  0.74895726, -0.62122729],
            [ 0.89300914, -0.41642107, -0.17066974]
        ])
        local_mapper.capture_frame(np.zeros(3,), R)

        # Run Segmenting
        print("Segment Frame")
        local_mapper.segment_frame()

        # Update Occ
        print("Update Occ")
        local_mapper.update_occ_grid(np.zeros((3,)))
        
        # Fit Free Space
        print("Fit Free")
        local_mapper.fit_free_space()

        print("Visualize Solution")
        grid_img = local_mapper.occ_grid
        grid_img[grid_img == 0] = 255                       # Free
        grid_img[(grid_img > 0) & (grid_img < 101)] = 0     # Occupied
        grid_img[grid_img < 0] = 150                        # Unknown
        grid_img = grid_img.astype(np.uint8)  # Assuming 1 = occupied, 0 = free
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlaying polytopes
        # grid_img = cv2.resize(grid_img, (grid_img.shape[0] * im_scale, grid_img.shape[0] * im_scale), interpolation=cv2.INTER_LINEAR)

        # Overlay polytopes
        for polytope in local_mapper.polytopes:
            vertices = polytope['vertices']
            vy, vx = local_mapper.xy_to_ind_(vertices)
            vertices = np.vstack((vx, vy)).T  # Ensure integer format for OpenCV
            if vertices.shape[0] > 2:  # Only plot if it's a valid polygon
                # cv2.polylines(grid_img, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue outline
                cv2.fillPoly(grid_img, [vertices], color=(255, 0, 0, 50))  # Semi-transparent fill

        # Display the image
        cv2.imshow(window_name, grid_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nRecording stopped.")

# Stop pipeline
local_mapper.shutdown()
