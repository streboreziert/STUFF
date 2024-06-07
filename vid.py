import cv2
import numpy as np

# Function to detect chessboard corners and calculate homography matrix
def detect_chessboard_and_homography(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the size of the chessboard
    chessboard_size = (7, 7)  # Change according to your chessboard
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Calculate homography matrix
        pts_src = np.zeros((chessboard_size[0] * chessboard_size[1], 2), dtype=np.float32)
        for i in range(chessboard_size[0]):
            for j in range(chessboard_size[1]):
                pts_src[i * chessboard_size[1] + j] = corners[i * chessboard_size[1] + j][0]

        pts_dst = np.zeros((chessboard_size[0] * chessboard_size[1], 2), dtype=np.float32)
        for i in range(chessboard_size[0]):
            for j in range(chessboard_size[1]):
                pts_dst[i * chessboard_size[1] + j] = np.array([j * 10, i * 10])

        homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
        
        return homography_matrix, image
    else:
        print("Chessboard corners not found in the image.")
        return None, None

# Paths to the two images (IR camera and RGB camera photos with chessboards)
ir_image_path = 'ir_image.jpg'
rgb_image_path = 'rgb_image.jpg'

# Detect chessboard corners and calculate homography matrix for the IR camera photo
homography_matrix_ir, ir_image = detect_chessboard_and_homography(ir_image_path)

# Detect chessboard corners and calculate homography matrix for the RGB camera photo
homography_matrix_rgb, rgb_image = detect_chessboard_and_homography(rgb_image_path)

# Check if both homography matrices are found
if homography_matrix_ir is not None and homography_matrix_rgb is not None:
    # Print coefficients of the homography matrix for the IR camera photo
    print("Homogeneous Matrix Transformation Coefficients for IR Camera Photo:")
    print(homography_matrix_ir)
    print("\n")

    # Print coefficients of the homography matrix for the RGB camera photo
    print("Homogeneous Matrix Transformation Coefficients for RGB Camera Photo:")
    print(homography_matrix_rgb)
    print("\n")

