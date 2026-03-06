import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("dataset/image11.jpg")
img2 = cv2.imread("dataset/image22.jpg")

# Check if loaded
if img1 is None or img2 is None:
    print("Error loading images")
    exit()

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Image1 Keypoints:", len(kp1))
print("Image2 Keypoints:", len(kp2))

# Draw keypoints
img1_kp = cv2.drawKeypoints(img1, kp1, None)
img2_kp = cv2.drawKeypoints(img2, kp2, None)

# Show images
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Image 1 Keypoints")
plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Image 2 Keypoints")
plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))

plt.show()
