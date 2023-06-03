import pytesseract
import cv2

# Set the language-specific model
lang_model = 'eng'  # Change to the appropriate language for your input image

# Set the OCR configuration settings
ocr_config = r"--psm 11 --oem 3 -l {}".format(lang_model)

# Load the image
img = cv2.imread("a.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to the grayscale image
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply a median filter to the thresholded image to remove noise
gray = cv2.medianBlur(gray, 3)

# Apply dilations and erosions to the thresholded image to improve OCR accuracy
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
gray = cv2.erode(gray, kernel, iterations=1)
gray = cv2.dilate(gray, kernel, iterations=1)

# Perform text detection to get the bounding boxes
text_boxes = pytesseract.image_to_boxes(gray, config=ocr_config)

# Draw the bounding boxes and OCR text on the image
for box in text_boxes.splitlines():
    box = box.split(" ")
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    cv2.rectangle(img, (x, img.shape[0] - y), (w, img.shape[0] - h), (0, 255, 0), 2)
    cv2.putText(img, box[0], (x, img.shape[0] - y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the image with bounding boxes and OCR text
cv2.imshow("img", img)
cv2.waitKey(0)