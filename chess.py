import cv2
import numpy as np

def find_chessboard_corners(image, pattern_size=(9, 6)):
    """
    Detect chessboard corners in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    return ret, corners

def align_images(image1, image2, pattern_size=(9, 6)):
    """
    Align image2 to image1 using chessboard corners.
    """
    # Detect chessboard corners in both images
    ret1, corners1 = find_chessboard_corners(image1, pattern_size)
    ret2, corners2 = find_chessboard_corners(image2, pattern_size)

    if not ret1 or not ret2:
        print("Chessboard corners not found in one or both images.")
        return None

    # Find the homography matrix
    H, status = cv2.findHomography(corners2, corners1)

    # Warp image2 to the perspective of image1
    height, width, channels = image1.shape
    aligned_image2 = cv2.warpPerspective(image2, H, (width, height))

    return aligned_image2

def blend_images(image1, image2, alpha=0.5):
    """
    Blend two images with the given transparency alpha.
    """
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended_image

def main(image1_path, image2_path, output_path, pattern_size=(9, 6), alpha=0.5):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Align the images
    aligned_image2 = align_images(image1, image2, pattern_size)

    if aligned_image2 is not None:
        # Blend the images
        blended_image = blend_images(image1, aligned_image2, alpha)

        # Save the result
        cv2.imwrite(output_path, blended_image)
        print(f"Blended image saved to {output_path}")
    else:
        print("Failed to align images.")

# Example usage
image1_path = 'path_to_image1.jpg'
image2_path = 'path_to_image2.jpg'
output_path = 'path_to_output_image.jpg'

main(image1_path, image2_path, output_path)
