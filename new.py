import cv2
import numpy as np

# Function to find chessboard corners
def find_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 8), None)
    return corners, gray

# Capture frames from cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
cap2 = cv2.VideoCapture(1)  # Camera 2

# Initialize transformation matrix
M = np.eye(3)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Find chessboard corners in both frames
    corners1, gray1 = find_corners(frame1)
    corners2, gray2 = find_corners(frame2)

    if corners1 is not None and corners2 is not None:
        # Compute transformation matrix
        M = cv2.findHomography(corners2, corners1)[0]

        # Warp frame2 to align with frame1
        aligned_frame2 = cv2.warpPerspective(frame2, M, (frame1.shape[1], frame1.shape[0]))

        # Blend the two frames
        blended_frame = cv2.addWeighted(frame1, 0.5, aligned_frame2, 0.5, 0)

        # Display the blended frame
        cv2.imshow('Blended Video', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture objects and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
