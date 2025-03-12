import cv2
import numpy as np
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = cv2.imread(r"M:\assignments\ambulance_detection\ambulance-car-isolated-on-white-260nw-1736917856.webp")
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Color detection (red, white, blue)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv, lower_red1, upper_red1)

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Combine masks
combined_mask = cv2.bitwise_or(red_mask, white_mask)
combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

# Find contours
contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
possible_ambulances = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if area > 5000 and w / h > 1.2:
        possible_ambulances.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Text detection
text_detected = False
for (x, y, w, h) in possible_ambulances:
    roi = image[y:y + h, x:x + w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh_roi, config='--psm 6')
    if any(word in text.upper() for word in ["AMBULANCE", "EMS", "RESCUE"]):
        text_detected = True
        cv2.putText(image, "AMBULANCE DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Siren detection
siren_detected = np.sum(blue_mask) > 5000

# Template matching for symbols
symbols = ["red_cross.png", "star_of_life.png"]
found_symbol = False
for symbol in symbols:
    template = cv2.imread(symbol, 0)
    if template is not None:
        res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)
        for pt in zip(*loc[::-1]):
            found_symbol = True
            cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (255, 0, 0), 2)

# Final decision
if (len(possible_ambulances) > 0 and (text_detected or found_symbol or siren_detected)):
    print("🚑 Ambulance Detected!")
    cv2.putText(image, "🚑 Ambulance Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
else:
    print("❌ No Ambulance Detected.")

# Display and save result
cv2.imshow("Ambulance Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", image)