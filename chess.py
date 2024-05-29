import cv2
import numpy as np

# Function to extract chessboard corners from frames
def extract_chessboard_corners(frames):
    chessboard_size = (7, 6)  # Size of the chessboard
    corners_list = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners_list.append(corners)
    return corners_list

# Function to calculate the transformation matrix
def calculate_transformation_matrix(corners_list1, corners_list2):
    object_points = np.zeros((len(corners_list1), 7 * 6, 3), np.float32)
    object_points[:, :, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    corners_list1 = np.array(corners_list1)
    corners_list2 = np.array(corners_list2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [corners_list1], gray.shape[::-1], None, None
    )

    # Calculate transformation matrix
    retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        object_points, corners_list2, mtx, dist
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    translation_matrix = np.eye(4)
    translation_matrix[0:3, 0:3] = rotation_matrix
    translation_matrix[0:3, 3] = translation_vector[:, 0]

    return translation_matrix

# Function to blend videos
def blend_videos(video1, video2, transform_matrix):
    out_width = int(video1.shape[1] * 1.5)
    out_height = video1.shape[0]

    # Create a blank canvas to blend videos
    blended_video = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # Overlay the first video
    blended_video[:, :video1.shape[1], :] = video1

    # Warp the second video
    warped_video = cv2.warpPerspective(video2, transform_matrix, (out_width, out_height))

    # Blend the warped video with the first video
    blended_video[:, video1.shape[1]:, :] = warped_video

    return blended_video

# Capture two live video streams
cap1 = cv2.VideoCapture(0)  # Video stream 1
cap2 = cv2.VideoCapture(1)  # Video stream 2

# Extract first 100 frames from each video
frames1 = [cap1.read()[1] for _ in range(100)]
frames2 = [cap2.read()[1] for _ in range(100)]

# Extract chessboard corners from frames
corners_list1 = extract_chessboard_corners(frames1)
corners_list2 = extract_chessboard_corners(frames2)

# Calculate transformation matrix
transformation_matrix = calculate_transformation_matrix(corners_list1, corners_list2)

# Merge videos
blended_video = blend_videos(frames1[0], frames2[0], transformation_matrix)

# Display original videos and aligned video
cv2.imshow('Original Video 1', frames1[0])
cv2.imshow('Original Video 2', frames2[0])
cv2.imshow('Aligned Video', blended_video)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Release video capture objects
cap1.release()
cap2.release()
