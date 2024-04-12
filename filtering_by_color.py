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

        # Display the mask
        cv2.imshow('mask', mask)

        # Apply bitwise AND between original image and mask to get filtered result
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours in the mask
        thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 16)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        # Fill rectangular contours
        for c in contours:
            cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

        # Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=8)

        # Draw rectangles
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(res, (x, y), (x + w, y + h), (36, 255, 12), 3)

        # Display the filtered result
        cv2.imshow('filtered_result', res)

    if key == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
