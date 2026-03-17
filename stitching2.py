import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("dataset/image33.jpg")
img2 = cv2.imread("dataset/image44.jpg")

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

# ------------------------------
# BF MATCHER + KNN
# ------------------------------

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# ------------------------------
# RATIO TEST
# ------------------------------

good_matches = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)
 
print("Total Matches:", len(matches))
print("Good Matches after ratio test:", len(good_matches))

# ------------------------------
# DRAW MATCHES
# ------------------------------

matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    # good_matches[:50],   # show first 50 matches
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(15,7))
plt.title("Feature Matching")
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
 
 
