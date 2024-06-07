import cv2
import numpy as np

# Function to detect chessboard corners and calculate homography matrix
def detect_chessboard_and_homography(gray_image):
    # Define the size of the chessboard
    chessboard_size = (7, 7)  # Change according to your chessboard
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    
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
        
        return homography_matrix
    else:
        print("Chessboard corners not found in the image.")
        return None

# Paths to the IR image and RGB image with chessboards
ir_image_path = 'ir_image.jpg'
rgb_image_path = 'rgb_image.jpg'

# Read the IR image
ir_image = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
if ir_image is None:
    print("Error: Could not open or find the IR image.")
else:
    # Detect chessboard corners and calculate homography matrix for the IR image
    homography_matrix_ir = detect_chessboard_and_homography(ir_image)

    # Read the RGB image
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        print("Error: Could not open or find the RGB image.")
    else:
        # Detect chessboard corners and calculate homography matrix for the RGB image
        homography_matrix_rgb = detect_chessboard_and_homography(rgb_image)

        # Check if both homography matrices are found
        if homography_matrix_ir is not None and homography_matrix_rgb is not None:
            # Print coefficients of the homography matrix for the IR image
            print("Homogeneous Matrix Transformation Coefficients for IR Image:")
            print(homography_matrix_ir)
            print("\n")

            # Print coefficients of the homography matrix for the RGB image
            print("Homogeneous Matrix Transformation Coefficients for RGB Image:")
            print(homography_matrix_rgb)
            print("\n")


 # Warp the IR image onto the RGB image using the homography matrix
            warped_ir_image = cv2.warpPerspective(ir_image_resized, homography_matrix_ir, (rgb_image.shape[1], rgb_image.shape[0]))

            # Make the IR image slightly transparent and overlay it on the RGB image
            alpha = 0.5  # Adjust transparency level (0.0 to 1.0)
            rgb_image_float = rgb_image.astype(float)
            warped_ir_image_color = cv2.cvtColor(warped_ir_image, cv2.COLOR_GRAY2BGR).astype(float)
            blended_image = cv2.addWeighted(rgb_image_float, alpha, warped_ir_image_color, 1 - alpha, 0)

            # Convert the blended image back to uint8 (required by imshow)
            blended_image = blended_image.astype(np.uint8)

            # Display the blended image
            cv2.imshow("Blended Image", blended_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
