import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("dataset/image2.jpg")
img2 = cv2.imread("dataset/image1.jpg")

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
 
print("Total Matches:", len(matches))
print("Good Matches after ratio test:", len(good_matches))

# -----------------------------------
# HOMOGRAPHY + RANSAC
# -----------------------------------

if len(good_matches) > 10:

    # Extract matched keypoints
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1,1,2)

    # Compute Homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Count inliers after RANSAC
    inliers = np.sum(mask)

    print("Matches after RANSAC (inliers):", inliers)
    print("Outliers removed:", len(good_matches) - inliers)

    print("Homography Matrix:\n", H)


    #optional

    # Draw only inlier matches
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    inlier_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15,7))
    plt.title("Matches After RANSAC (Inliers)")
    plt.imshow(cv2.cvtColor(inlier_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

 
    # -----------------------------------
    # WARP IMAGE (Correct panorama bounds)
    # -----------------------------------

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of the images
    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    # Transform img1 corners using homography
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # Combine corners
    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

    # Find min and max coordinates
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation to avoid negative coordinates
    translation = [-xmin, -ymin]

    # Translation matrix
    T = np.array([
        [1,0,translation[0]],
        [0,1,translation[1]],
        [0,0,1]
    ])

    # Warp image1
    result = cv2.warpPerspective(img1, T @ H, (xmax-xmin, ymax-ymin))

    # Place image2 into panorama
    result[translation[1]:h2+translation[1],
           translation[0]:w2+translation[0]] = img2

    

    # Show stitched result
    plt.figure(figsize=(12,6))
    plt.title("Stitched Image")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

else:
    print("Not enough matches found!")

 
 
 
