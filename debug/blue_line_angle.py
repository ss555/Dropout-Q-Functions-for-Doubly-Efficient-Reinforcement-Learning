import cv2
import numpy as np

# Load image
image = cv2.imread('path_to_your_image.jpg')

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for blue color and apply mask
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Edge detection
edges = cv2.Canny(mask, 50, 150)

# Line detection
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Calculate and print angle with horizontal
for line in lines:
    rho, theta = line[0]
    angle = (theta * 180.0 / np.pi) - 90 # Convert to degrees and adjust to horizontal
    print(f'Angle with horizontal: {angle} degrees')