import cv2
import numpy as np

# Function to blend frames with transparency
def blend_with_transparency(frame1, frame2, alpha):
    blended_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
    return blended_frame

# Capture frames from cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
cap2 = cv2.VideoCapture(1)  # Camera 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize frame2 to match the size of frame1
    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Set alpha value for transparency (adjust as needed)
    alpha = 0.5

    # Blend frames with transparency
    blended_frame = blend_with_transparency(frame1, frame2, alpha)

    # Display the blended frame
    cv2.imshow('Blended Video', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture objects and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
