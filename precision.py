import cv2
import numpy as np

# Function to detect lines in a frame using Hough Line Transform
def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines

# Function to get corner points from the detected lines
def get_corner_points(lines, frame_shape):
    corners = []

    # Ensure we have detected some lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 != x2:  # Avoid vertical lines to prevent division by zero
                slope = (y2 - y1) / (x2 - x1)
                if -0.5 < slope < 0.5:  # Filter out near-horizontal lines
                    if x1 < frame_shape[1] / 2 and x2 < frame_shape[1] / 2:
                        corners.append((x1, y1))
                    elif x1 > frame_shape[1] / 2 and x2 > frame_shape[1] / 2:
                        corners.append((x2, y2))
    
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

    # Detect lines in both frames
    lines1 = detect_lines(frame1)
    lines2 = detect_lines(frame2)

    # Get corner points from the detected lines
    corners1 = get_corner_points(lines1, frame1.shape)
    corners2 = get_corner_points(lines2, frame2.shape)

    if corners1 and corners2:
        # Estimate transformation matrix using RANSAC
        M, _ = cv2.findHomography(np.array(corners2), np.array(corners1), cv2.RANSAC, 5.0)

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
