import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("dataset/img2.jpg")   # left image
img2 = cv2.imread("dataset/img1.jpg")   # right image

# ------------------------------
# SHOW INPUT IMAGES
# ------------------------------

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Input Image 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Input Image 2")
plt.axis("off")

plt.show()

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
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(15,7))
plt.title("Feature Matching Between Images")
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


# -----------------------------------
# HOMOGRAPHY + RANSAC
# -----------------------------------

if len(good_matches) > 10:

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = np.sum(mask)

    print("Matches after RANSAC (inliers):", inliers)
    print("Outliers removed:", len(good_matches) - inliers)

    print("Homography Matrix:\n", H)


    # -----------------------------------
    # WARP IMAGE (Panorama bounds)
    # -----------------------------------

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]

    T = np.array([
        [1,0,translation[0]],
        [0,1,translation[1]],
        [0,0,1]
    ])

    # Warp image
    warped_img1 = cv2.warpPerspective(img1, T @ H, (xmax-xmin, ymax-ymin))

    # Place image2
    panorama = warped_img1.copy()
    panorama[
        translation[1]:h2+translation[1],
        translation[0]:w2+translation[0]
    ] = img2


    # ------------------------------
    # IMPROVED CROP: LARGEST INTERNAL RECTANGLE
    # ------------------------------

    # 1. Create a mask of the panorama (black = 0, image = 255)
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 2. Find the largest external contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    # 3. Use an iterative approach to shrink the bounding box
    # until it contains ONLY non-zero pixels
    x, y, w, h = cv2.boundingRect(c)

    while True:
        # Check if the current rectangle contains any black pixels
        # We look at the mask (thresh) within the current coordinates
        sub_mask = thresh[y:y+h, x:x+w]
        
        # If the sub_mask is entirely white (255), we found our crop!
        if np.all(sub_mask > 0):
            break
        
        # Otherwise, shrink the box from all sides by 1 pixel and check again
        # (You can also optimize this by checking which side has more black pixels)
        x += 1
        y += 1
        w -= 2
        h -= 2
        
        # Safety break to avoid infinite loops
        if w <= 0 or h <= 0:
            break

    cropped = panorama[y:y+h, x:x+w]


    # ------------------------------
    # SHOW FINAL PANORAMA
    # ------------------------------

    plt.figure(figsize=(12,6))
    plt.title("Final Panorama (Black Borders Removed)")
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

else:
    print("Not enough matches found!")
