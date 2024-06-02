import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Define the chessboard size (number of inner corners per chessboard row and column)
chessboard_size = (7, 7)  # Change this to the actual size of your chessboard

# Find the chessboard corners in both images
ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

if ret1 and ret2:
    # Refine corner positions
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Compute the homography matrix
    H, mask = cv2.findHomography(corners1, corners2, cv2.RANSAC)

    print("Homography matrix:")
    for row in H:
        for coefficient in row:
            print(f"{coefficient:.6f}", end=' ')
        print()
else:
    print("Chessboard couldn't be detected in one or both images.")

# Optionally, visualize the detected corners
img1_corners = cv2.drawChessboardCorners(img1, chessboard_size, corners1, ret1)
img2_corners = cv2.drawChessboardCorners(img2, chessboard_size, corners2, ret2)

cv2.imshow('Corners in Image 1', img1_corners)
cv2.imshow('Corners in Image 2', img2_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

