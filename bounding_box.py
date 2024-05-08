import cv2
import numpy as np
from coordinates import get_bay_coordinates

# Initialize VideoCapture
cap = cv2.VideoCapture('rtsp://admin:iith@123@192.168.74.16:554/Streaming/Channels/101')

# Get the parking bay coordinates
bay_coordinates = get_bay_coordinates()

# Polygon properties
isClosed = True
color = (34, 65, 200)  # BGR format, change the color if needed
thickness = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for display
    frame = cv2.resize(frame, (960, 540))

    # Draw bounding boxes (polygons) on the frame
    for bay in bay_coordinates:
        pts = np.array(bay, np.int32)
        pts = pts.reshape((-1, 1, 2))  # Ensure correct shape
        cv2.polylines(frame, [pts], isClosed, color, thickness)

    # Display the frame
    cv2.imshow("Capturing", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
