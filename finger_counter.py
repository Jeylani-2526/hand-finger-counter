import cv2
import numpy as np

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror-like view
    frame = cv2.flip(frame, 1)

    # Define ROI (Region of Interest) for hand detection
    roi = frame[100:400, 350:650]
    cv2.rectangle(frame, (350, 100), (650, 400), (0, 255, 0), 2)

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Draw contour on ROI
        cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)

        # Optionally draw bounding box
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the frames
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Skin Mask", mask)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
