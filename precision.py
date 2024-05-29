import cv2
import numpy as np

# Function to detect corners using Shi-Tomasi corner detection
def detect_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    return corners

# Function to match features using ORB (or any other feature matching algorithm)
def match_features(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

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

    if len(corners1) > 0 and len(corners2) > 0:
        # Estimate transformation matrix using RANSAC
        M, _ = cv2.findHomography(corners2, corners1, cv2.RANSAC, 5.0)

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
