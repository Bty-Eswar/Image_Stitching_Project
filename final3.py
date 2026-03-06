import cv2
import numpy as np
import matplotlib.pyplot as plt

"""def crop_black_borders(image):
    
    Iteratively removes black borders (pixels = 0) from the stitched image.
    This fixes the 'black region' issue caused by warping/rotation.
    
    # Convert to gray to check for black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the largest contour to get the initial bounding rect
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # Get bounding box of the non-black area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop to the bounding box first
    crop = image[y:y+h, x:x+w]
    thresh = thresh[y:y+h, x:x+w]

    # Iteratively shave off rows/cols that still have black pixels
    # Top
    while thresh.shape[0] > 0 and np.any(thresh[0, :] == 0):
        thresh = thresh[1:, :]
        crop = crop[1:, :]
        
    # Bottom
    while thresh.shape[0] > 0 and np.any(thresh[-1, :] == 0):
        thresh = thresh[:-1, :]
        crop = crop[:-1, :]
        
    # Left
    while thresh.shape[1] > 0 and np.any(thresh[:, 0] == 0):
        thresh = thresh[:, 1:]
        crop = crop[:, 1:]
        
    # Right
    while thresh.shape[1] > 0 and np.any(thresh[:, -1] == 0):
        thresh = thresh[:, :-1]
        crop = crop[:, :-1]
        
    return crop"""
def safe_crop_panorama(image):
    """
    Crops the image to the bounding box of the non-black area.
    This guarantees you won't lose the image, even if it's rotated.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold to separate image from black background
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No content found to crop!")
        return image

    # 4. Find the largest contour (the actual panorama)
    c = max(contours, key=cv2.contourArea)
    
    # 5. Get the bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(c)
    
    # 6. Crop the image
    cropped_result = image[y:y+h, x:x+w]
    
    return cropped_result
# ---------------------------------------------------------
# 1. LOAD IMAGES
# ---------------------------------------------------------
# Make sure these paths are correct for your folder structure
img1 = cv2.imread("dataset/img2.jpg")   # Left image
img2 = cv2.imread("dataset/img1.jpg")   # Right image

if img1 is None or img2 is None:
    print("Error: Images not found! Check your paths.")
    exit()

# Convert to grayscale for SIFT
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------
# 2. FEATURE DETECTION (SIFT) [cite: 15, 211]
# ---------------------------------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"Keypoints - Image 1: {len(kp1)}, Image 2: {len(kp2)}")

# ---------------------------------------------------------
# 3. FEATURE MATCHING (BF Matcher + KNN) [cite: 17, 262]
# ---------------------------------------------------------
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Ratio Test (Lowe's Ratio Test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Good Matches after ratio test: {len(good_matches)}")

# Visualization of matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# ---------------------------------------------------------
# 4. HOMOGRAPHY & WARPING [cite: 54, 298]
# ---------------------------------------------------------
if len(good_matches) > 10:
    # Get points from matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate Homography with RANSAC [cite: 18, 276]
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # -----------------------------------------------------
    # CALCULATE CANVAS SIZE
    # -----------------------------------------------------
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of img1 (the one being transformed)
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Get corners of img2 (the reference image)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Transform corners of img1
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    
    # Combine all corners to find total size
    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
    
    # Translation matrix to shift the image into positive coordinates
    translation_dist = [-xmin, -ymin]
    T = np.array([[1, 0, translation_dist[0]], 
                  [0, 1, translation_dist[1]], 
                  [0, 0, 1]])
    
    # Warp img1
    warped_img1 = cv2.warpPerspective(img1, T @ H, (xmax - xmin, ymax - ymin))
    
    # Create a canvas for img2 and place it
    warped_img2 = np.zeros_like(warped_img1)
    warped_img2[translation_dist[1]:h2+translation_dist[1], 
                translation_dist[0]:w2+translation_dist[0]] = img2

    # -----------------------------------------------------
    # 5. BLENDING (Alpha Blending) [cite: 19, 315]
    # -----------------------------------------------------
    # Create masks to find where images exist
    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0
    
    # Where both exist (overlap), average them (Alpha = 0.5)
    overlap = mask1 & mask2
    
    # Initialize final panorama
    panorama = warped_img1.copy()
    
    # Place img2 where img1 doesn't exist
    panorama[~mask1] = warped_img2[~mask1]
    
    # Blend the overlap area
    panorama[overlap] = (warped_img1[overlap] * 0.5 + warped_img2[overlap] * 0.5).astype(np.uint8)

    # -----------------------------------------------------
    # 6. REMOVE BLACK REGIONS (Cropping) 
    # -----------------------------------------------------
    final_output = safe_crop_panorama(panorama)
    # -----------------------------------------------------
    # 7. DISPLAY RESULTS
    # -----------------------------------------------------
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Image 1 (Left)")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Image 2 (Right)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Matches ({len(good_matches)})")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama (Cropped)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("Not enough matches found to stitch!")