import cv2
import os

# Create a directory to save images if it doesn't exist
output_dir = 'captured_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set frame counter
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window
    cv2.imshow('Live Video', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save the frame if 's' key is pressed
        img_name = os.path.join(output_dir, f'frame_{frame_count}.png')
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        frame_count += 1

    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
