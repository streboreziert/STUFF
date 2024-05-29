import cv2
import numpy as np

# Function to detect corners in a frame
def detect_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    return corners

# Function to refine corners using subpixel accuracy
def refine_corners(frame, corners):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners

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

    # Resize frame2 to match the size of frame1
    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Detect corners in both frames
    corners1 = detect_corners(frame1)
    corners2 = detect_corners(frame2)

    if corners1 is not None and corners2 is not None:
        # Refine corners using subpixel accuracy
        corners1_refined = refine_corners(frame1, corners1)
        corners2_refined = refine_corners(frame2, corners2)

        # Estimate transformation matrix using RANSAC
        M, _ = cv2.findHomography(corners2_refined, corners1_refined, cv2.RANSAC, 5.0)

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
