import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD IMAGES
# ------------------------------
img1 = cv2.imread("dataset/image1.jpg")   # Left image
img2 = cv2.imread("dataset/image2.jpg")   # Right image

# Check if images loaded
if img1 is None or img2 is None:
    print("Error loading images. Check file paths.")
    exit()

# Convert to grayscale for detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. DETECT FEATURES (SIFT)
# ------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# --- PRINT KEYPOINTS ---
print("Image1 Keypoints:", len(kp1))
print("Image2 Keypoints:", len(kp2))

# 3. MATCH FEATURES (BF Matcher + KNN)
# ------------------------------
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# --- PRINT TOTAL MATCHES ---
print("Total Matches:", len(matches))

# Apply Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# --- PRINT GOOD MATCHES ---
print("Good Matches after ratio test:", len(good_matches))

# Create the "Matches" image
matched_img = cv2.drawMatches(
    img1, kp1, 
    img2, kp2, 
    good_matches, 
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 4. HOMOGRAPHY & WARPING
# ------------------------------
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # --- PRINT HOMOGRAPHY INFO ---
    inliers = np.sum(mask)
    print("Matches after RANSAC (inliers):", inliers)
    print("Outliers removed:", len(good_matches) - inliers)
    print("Homography Matrix:\n", H)

    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners to calculate canvas size
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    
    T = np.array([[1, 0, translation[0]], 
                  [0, 1, translation[1]], 
                  [0, 0, 1]])

    # Warp and Stitch
    warped_img1 = cv2.warpPerspective(img1, T @ H, (xmax - xmin, ymax - ymin))
    panorama = warped_img1.copy()
    panorama[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = img2

    '''# 5. ROBUST CROP (Iterative Shaving)
    # ------------------------------
    gray_pano = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_pano, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        crop = panorama[y:y+h, x:x+w]
        thresh_crop = thresh[y:y+h, x:x+w]
        
        # Top
        while np.any(thresh_crop[0, :] == 0) and thresh_crop.shape[0] > 0:
            thresh_crop = thresh_crop[1:, :]
            crop = crop[1:, :]
            
        # Bottom
        while np.any(thresh_crop[-1, :] == 0) and thresh_crop.shape[0] > 0:
            thresh_crop = thresh_crop[:-1, :]
            crop = crop[:-1, :]
            
        # Left
        while np.any(thresh_crop[:, 0] == 0) and thresh_crop.shape[1] > 0:
            thresh_crop = thresh_crop[:, 1:]
            crop = crop[:, 1:]
            
        # Right
        while np.any(thresh_crop[:, -1] == 0) and thresh_crop.shape[1] > 0:
            thresh_crop = thresh_crop[:, :-1]
            crop = crop[:, :-1]
            
        cropped_panorama = crop
    else:
        cropped_panorama = panorama
        

    '''# 5. ROBUST CROP (Fixed)
    # ------------------------------
    gray_pano = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_pano, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        crop = panorama[y:y+h, x:x+w]
        thresh_crop = thresh[y:y+h, x:x+w]
        
        # NOTE: We swapped the order (check shape > 0 FIRST)
        
        # Top
        while thresh_crop.shape[0] > 0 and np.any(thresh_crop[0, :] == 0):
            thresh_crop = thresh_crop[1:, :]
            crop = crop[1:, :]
            
        # Bottom
        while thresh_crop.shape[0] > 0 and np.any(thresh_crop[-1, :] == 0):
            thresh_crop = thresh_crop[:-1, :]
            crop = crop[:-1, :]
            
        # Left
        while thresh_crop.shape[1] > 0 and np.any(thresh_crop[:, 0] == 0):
            thresh_crop = thresh_crop[:, 1:]
            crop = crop[:, 1:]
            
        # Right
        while thresh_crop.shape[1] > 0 and np.any(thresh_crop[:, -1] == 0):
            thresh_crop = thresh_crop[:, :-1]
            crop = crop[:, :-1]
            
        # SAFETY CHECK: If the crop deleted everything, return the uncropped version
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            print("Cropping failed (image too distorted). Returning uncropped result.")
            cropped_panorama = panorama[y:y+h, x:x+w] # Just return the bounding box
        else:
            cropped_panorama = crop
    else:
        cropped_panorama = panorama

    # 6. VISUALIZATION
    # ------------------------------
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Input Image 1 (Left)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Input Image 2 (Right)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Keypoint Matches ({len(good_matches)} good matches)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(cropped_panorama, cv2.COLOR_BGR2RGB))
    plt.title("Final Stitched Panorama")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("Not enough matches found - cannot stitch!")
