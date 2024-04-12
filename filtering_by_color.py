import cv2
import numpy as np

lower_hue = None
upper_hue = None

def on_click(event, x, y, flags, param):
    global lower_hue, upper_hue
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert the BGR image to HSV for easier color filtering
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extract the hue value from the HSV representation at the clicked point
        hue_value = hsv_frame[y][x]

        # Create a mask where pixels with similar hues are white (255) and others are black (0)
        lower_hue = np.array([max(hue_value[0] - 20, 0), max(hue_value[1] - 100, 0), max(hue_value[2] - 30, 0)])
        upper_hue = np.array([min(hue_value[0] + 20, 180), min(hue_value[1] + 100, 255), min(hue_value[2] + 30, 255)])

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', on_click)

    key = cv2.waitKey(1) & 0xFF

    if lower_hue is not None:
        # Convert the BGR image to HSV for easier color filtering
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply thresholding operation based on the selected color range
        mask = cv2.inRange(hsv_frame, lower_hue, upper_hue)

        # Apply bitwise AND between original image and mask to get filtered result
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the filtered result
        cv2.imshow('filtered_result', res)

    if key == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()