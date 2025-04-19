import cv2
import numpy as np
from collections import deque

# Initialize webcam
cap = cv2.VideoCapture(0)

# Finger count smoothing
counter_memory = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 3000:
            # Convex hulls
            hull_points = cv2.convexHull(max_contour)  # For drawing
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)  # For defects

            # Draw contours
            cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)
            cv2.drawContours(roi, [hull_points], -1, (0, 255, 255), 2)

            # Convexity Defects
            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(max_contour, hull_indices)

                if defects is not None:
                    count_defects = 0

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # Calculate angle using cosine rule
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

                        # If angle is less than 90° and depth is sufficient
                        if angle <= np.pi / 2 and d > 10000:
                            count_defects += 1
                            cv2.circle(roi, far, 5, (0, 0, 255), -1)

                    # Final finger count (defects + 1)
                    finger_count = count_defects + 1
                    counter_memory.append(finger_count)

                    # Stabilize count using most frequent recent value
                    stable_count = max(set(counter_memory), key=counter_memory.count)
                    # Choose gesture label and emoji
                    gesture = {
                        0: "Fist",
                        1: "One",
                        2: "Peace",
                        3: "Rock",
                        4: "Vulcan",
                        5: "Palm"
                    }

                    label = gesture.get(stable_count)

                    # Display emoji + gesture label
                    cv2.putText(frame, f"{label}", (250, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 200), 3)

                    # Display result
                    cv2.putText(frame, f"Fingers: {stable_count}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                    # UI overlay
                    cv2.putText(frame, "Place your hand inside the green box", (30, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.putText(frame, "Press 'q' to quit", (30, 490),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    cv2.putText(frame, "Detecting fingers...", (350, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # Display
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Skin Mask", mask)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
