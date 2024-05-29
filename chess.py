import cv2
import numpy as np

# Read the two images
image1 = cv2.imread('chessboard1.jpg')
image2 = cv2.imread('chessboard2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Set the chessboard size (number of inner corners per chessboard row and column)
chessboard_size = (7, 7)  # for a 8x8 chessboard, the inner corners are 7x7

# Find the chessboard corners in both images
ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

if ret1 and ret2:
    # Refine the corner positions
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), 
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), 
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Find the homography matrix
    H, status = cv2.findHomography(corners1, corners2, cv2.RANSAC)

    # Warp the first image to align with the second image
    height, width, channels = image2.shape
    aligned_image1 = cv2.warpPerspective(image1, H, (width, height))

    # Show the aligned image
    cv2.imshow('Aligned Image 1', aligned_image1)
    cv2.imshow('Image 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not found in one or both images.")
