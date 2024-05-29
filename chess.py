import cv2
import numpy as np

# Function to detect and draw chessboard corners
def find_and_draw_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 8), None)

    if ret:
        cv2.drawChessboardCorners(frame, (8, 8), corners, ret)
        return corners
    else:
        return None

# Function to align frames using perspective transformation
def align_frames(frame1, frame2, corners1, corners2):
    h, w = frame1.shape[:2]

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners2, corners1)

    # Apply the transformation to frame2 to align it with frame1
    aligned_frame = cv2.warpPerspective(frame2, M, (w, h))

    return aligned_frame

# Function to blend two frames together
def blend_frames(frame1, frame2, alpha=0.5):
    blended_frame = cv2.addWeighted(frame1, alpha, frame2, 1-alpha, 0)
    return blended_frame

# Capture frames from cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
cap2 = cv2.VideoCapture(1)  # Camera 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Find chessboard corners in both frames
    corners1 = find_and_draw_corners(frame1)
    corners2 = find_and_draw_corners(frame2)

    if corners1 is not None and corners2 is not None:
        # Align frames
        aligned_frame2 = align_frames(frame1, frame2, corners1, corners2)

        # Blend frames
        blended_frame = blend_frames(frame1, aligned_frame2)

        # Display the blended frame
        cv2.imshow('Blended Video', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture objects and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
