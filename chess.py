import cv2
import numpy as np

# Paths to the images
rgb_image_paths = [
    'path_to_rgb_image1.png',
    'path_to_rgb_image2.png',
    'path_to_rgb_image3.png',
    'path_to_rgb_image4.png',
    'path_to_rgb_image5.png'
]

ir_image_paths = [
    'path_to_ir_image1.png',
    'path_to_ir_image2.png',
    'path_to_ir_image3.png',
    'path_to_ir_image4.png',
    'path_to_ir_image5.png'
]

# Chessboard dimensions
chessboard_size = (9, 6)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints_rgb = []  # 2d points in RGB image plane
imgpoints_ir = []  # 2d points in IR image plane

# Process each pair of images
for rgb_img_path, ir_img_path in zip(rgb_image_paths, ir_image_paths):
    rgb_img = cv2.imread(rgb_img_path)
    ir_img = cv2.imread(ir_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert IR image
    ir_img = cv2.bitwise_not(ir_img)
    
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    ret_rgb, corners_rgb = cv2.findChessboardCorners(gray_rgb, chessboard_size, None)
    ret_ir, corners_ir = cv2.findChessboardCorners(ir_img, chessboard_size, None)
    
    if ret_rgb and ret_ir:
        objpoints.append(objp)
        
        corners_rgb_refined = cv2.cornerSubPix(gray_rgb, corners_rgb, (11, 11), (-1, -1), criteria)
        corners_ir_refined = cv2.cornerSubPix(ir_img, corners_ir, (11, 11), (-1, -1), criteria)
        
        imgpoints_rgb.append(corners_rgb_refined)
        imgpoints_ir.append(corners_ir_refined)

# Compute the homography matrix using all the points
all_pts_rgb = np.vstack(imgpoints_rgb)
all_pts_ir = np.vstack(imgpoints_ir)

H, _ = cv2.findHomography(all_pts_ir, all_pts_rgb, cv2.RANSAC)

# Enhance the matrix by re-computing with refined points if necessary
# (In this example, we assume the initial computation is sufficient)

# Load a sample image to apply the transformation
sample_rgb_img = cv2.imread(rgb_image_paths[0])
sample_ir_img = cv2.imread(ir_image_paths[0], cv2.IMREAD_GRAYSCALE)
sample_ir_img = cv2.bitwise_not(sample_ir_img)  # Invert the sample IR image

# Apply the homography to the IR image
height, width = sample_rgb_img.shape[:2]
warped_ir = cv2.warpPerspective(sample_ir_img, H, (width, height))

# Convert the RGB image to green and IR image to red
green_rgb = cv2.cvtColor(sample_rgb_img, cv2.COLOR_BGR2GRAY)
green_rgb = cv2.merge([np.zeros_like(green_rgb), green_rgb, np.zeros_like(green_rgb)])

red_ir = cv2.merge([warped_ir, np.zeros_like(warped_ir), np.zeros_like(warped_ir)])

# Overlay the images
overlay = cv2.addWeighted(green_rgb, 0.5, red_ir, 0.5, 0)

# Display the result
cv2.imshow('Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print out the homography matrix
print("Homography Matrix:")
print(H)

