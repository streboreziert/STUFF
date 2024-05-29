import cv2
import numpy as np

# Define the termination criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Capture video streams
video1 = cv2.VideoCapture(0)  # Assuming the first camera
video2 = cv2.VideoCapture(1)  # Assuming the second camera

while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    
    if not (ret1 and ret2):
        break
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, (8,6), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (8,6), None)
    
    if ret1 and ret2:
        objpoints.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
        imgpoints.append(corners1)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame1, (8,6), corners1, ret1)
        cv2.drawChessboardCorners(frame2, (8,6), corners2, ret2)
        
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray1.shape[::-1], None, None)
        
        # Find the homography
        H, _ = cv2.findHomography(objp, corners2, cv2.RANSAC)
        
        # Warp the second frame to align with the first frame
        warped_frame2 = cv2.warpPerspective(frame2, H, (frame1.shape[1], frame1.shape[0]))
        
        # Blend the frames
        blended_frame = cv2.addWeighted(frame1, 0.5, warped_frame2, 0.5, 0)
        
        # Display the result
        cv2.imshow('Blended Video', blended_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects
video1.release()
video2.release()
cv2.destroyAllWindows()
