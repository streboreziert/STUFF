import cv2

# Open the video file or capture device
video_path = 'path_to_your_video.mp4'  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame counter
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not grabbed, we've reached the end of the video
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Video Frame', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the current frame as an image
        image_path = f'saved_image_{frame_count}.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Frame {frame_count} saved as {image_path}")

    # Increment frame counter
    frame_count += 1

    # Exit loop if 'q' key is pressed
    if key == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
