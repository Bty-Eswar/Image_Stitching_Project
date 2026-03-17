import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func


# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def largest_interior_rectangle(mask):
    """
    Find the largest rectangle that fits entirely within the non-black region.
    Uses a histogram-based approach (largest rectangle in histogram).
    Eliminates ALL black borders completely.
    """
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)  # (x, y, w, h)
    best_area = 0

    for row in range(h):
        for col in range(w):
            heights[col] = heights[col] + 1 if mask[row, col] else 0

        stack = []
        for col in range(w + 1):
            current_h = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] > current_h:
                s_col, s_h = stack.pop()
                area = s_h * (col - s_col)
                if area > best_area:
                    best_area = area
                    best = (s_col, row - s_h + 1, col - s_col, s_h)
                start = s_col
            stack.append((start, current_h))

    return best  # (x, y, w, h)


def feathered_blend(img1, img2, mask1, mask2):
    """
    Feathered (distance-transform) blending in the overlap region.
    Produces a smooth, seamless transition — eliminates visible seams.
    This extends the paper's alpha blending (Section 4.4) for better output quality.
    """
    overlap = mask1 & mask2

    dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)

    total = dist1 + dist2
    total[total == 0] = 1  # avoid division by zero
    alpha = dist1 / total  # weight for img1

    panorama = np.zeros_like(img1, dtype=np.float32)

    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    panorama[only1] = img1[only1]
    panorama[only2] = img2[only2]

    # Blended overlap: I = α*I1 + (1-α)*I2  [Paper Equation 6]
    for c in range(3):
        a = alpha[overlap]
        panorama[overlap, c] = (img1[overlap, c] * a +
                                img2[overlap, c] * (1 - a))

    return panorama.astype(np.uint8)


# =============================================================
# SECTION 5: PERFORMANCE EVALUATION METRICS
# =============================================================

def compute_mse(img1, img2):
    """Equation (14) — Mean Squared Error. Lower = better."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)

def compute_psnr(img1, img2):
    """Equation (7) — Peak Signal to Noise Ratio. Higher = better (30–50 dB acceptable)."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)

def compute_rmse(img1, img2):
    """Equation (9) — Root Mean Square Error. Lower = better."""
    return np.sqrt(compute_mse(img1, img2))

def compute_ssim(img1, img2):
    """Equation (8) — Structural Similarity Index. Higher = better (max = 1.0)."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_func(g1, g2, full=True)
    return score

def compute_uiqi(img1, img2):
    """
    Equation (13) — Universal Image Quality Index.
    UIQI = (4 * mu_x * mu_y * sigma_xy) / ((mu_x^2 + mu_y^2)(sigma_x^2 + sigma_y^2))
    Higher = better (max = 1.0). Computed per channel, then averaged.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    uiqi_channels = []
    for c in range(3):
        x = img1[:, :, c]
        y = img2[:, :, c]
        mu_x    = np.mean(x)
        mu_y    = np.mean(y)
        sigma_x2  = np.var(x)
        sigma_y2  = np.var(y)
        sigma_xy  = np.mean((x - mu_x) * (y - mu_y))
        denom = (mu_x**2 + mu_y**2) * (sigma_x2 + sigma_y2)
        uiqi_channels.append((4 * mu_x * mu_y * sigma_xy) / denom if denom != 0 else 1.0)
    return float(np.mean(uiqi_channels))

def evaluate_all_metrics(reference, stitched):
    """
    Resize stitched image to match reference, then compute all 5 paper metrics.
    Uses img2 (right/reference image) as the ground truth, consistent with paper Table 1.
    """
    ref = cv2.resize(reference, (stitched.shape[1], stitched.shape[0]))
    return {
        "PSNR (dB)" : compute_psnr(ref, stitched),
        "SSIM"      : compute_ssim(ref, stitched),
        "RMSE"      : compute_rmse(ref, stitched),
        "MSE"       : compute_mse(ref, stitched),
        "UIQI"      : compute_uiqi(ref, stitched),
    }


# =============================================================
# SECTION 4.1 — PREPROCESSING: Load images & convert to grayscale
# =============================================================
img1 = cv2.imread("dataset/image33.jpg")   # Left image
img2 = cv2.imread("dataset/image44.jpg")   # Right image

if img1 is None or img2 is None:
    print("Error: Images not found! Check your paths.")
    exit()

# Preprocessing: convert color images to grayscale (Paper Section 4.1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# =============================================================
# SECTION 4.1 — FEATURE DETECTION: SIFT
# Produces 128-dimensional descriptors (Paper Section 4.1)
# =============================================================
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"Keypoints — Image 1: {len(kp1)}, Image 2: {len(kp2)}")


# =============================================================
# SECTION 4.2 — FEATURE MATCHING: BF Matcher + KNN
# k=2 neighbors, ratio test threshold = 0.8 (Paper Section 4.2)
# =============================================================
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)   # k=2 as per paper

# Lowe's Ratio Test — threshold = 0.8 (Paper Section 4.2)
good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

print(f"Good matches after ratio test (threshold=0.8): {len(good_matches)}")

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)


# =============================================================
# SECTION 4.3 — RANSAC: Outlier Removal & Homography Estimation
# =============================================================
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC eliminates outliers and estimates homography H (Paper Section 4.3)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inliers = int(mask.sum())
    print(f"RANSAC inliers: {inliers} / {len(good_matches)}")


    # =============================================================
    # SECTION 4.4 — IMAGE WARPING: Perspective Warping
    # =============================================================
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners    = np.concatenate((warped_corners, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation_dist = [-xmin, -ymin]
    T = np.array([[1, 0, translation_dist[0]],
                  [0, 1, translation_dist[1]],
                  [0, 0, 1]])

    canvas_w, canvas_h = xmax - xmin, ymax - ymin

    # Perspective warp img1 onto canvas (Paper Section 4.4)
    warped_img1 = cv2.warpPerspective(img1, T @ H, (canvas_w, canvas_h))

    # Place img2 on canvas
    warped_img2 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_img2[translation_dist[1]:h2+translation_dist[1],
                translation_dist[0]:w2+translation_dist[0]] = img2


    # =============================================================
    # SECTION 4.4 — BLENDING: Feathered Alpha Blending
    # Implements equation (6): I = α*I1 + (1-α)*I2
    # Uses distance-transform weights for seamless transition
    # =============================================================
    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0

    panorama = feathered_blend(warped_img1, warped_img2, mask1, mask2)


    # =============================================================
    # CROP: Largest Interior Rectangle (removes all black borders)
    # =============================================================
    combined_mask = (cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) > 0)
    x, y, w, h = largest_interior_rectangle(combined_mask)

    if w > 0 and h > 0:
        final_output = panorama[y:y+h, x:x+w]
        print(f"Cropped to interior rectangle: x={x}, y={y}, w={w}, h={h}")
    else:
        contours, _ = cv2.findContours(
            combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        final_output = panorama[by:by+bh, bx:bx+bw]
        print("Fallback bounding-box crop used.")


    # =============================================================
    # SECTION 5 — PERFORMANCE EVALUATION (Paper Table 1)
    # =============================================================
    metrics = evaluate_all_metrics(img2, final_output)

    print("\n" + "="*50)
    print("  PERFORMANCE EVALUATION METRICS  (Paper Table 1)")
    print("="*50)
    print(f"  PSNR  : {metrics['PSNR (dB)']:.2f} dB   (higher = better, target 30–50 dB)")
    print(f"  SSIM  : {metrics['SSIM']:.4f}      (higher = better, max = 1.0)")
    print(f"  RMSE  : {metrics['RMSE']:.2f}        (lower  = better)")
    print(f"  MSE   : {metrics['MSE']:.2f}      (lower  = better)")
    print(f"  UIQI  : {metrics['UIQI']:.4f}      (higher = better, max = 1.0)")
    print("="*50)


    # =============================================================
    # DISPLAY — matches paper Fig. 4 layout:
    # (a) Input Image 1  (b) Input Image 2
    # (c) Matching Feature Points  (d) Final Output
    # =============================================================
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("(a) Input Image 1")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("(b) Input Image 2")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"(c) Matching Feature Points ({len(good_matches)} matches, "
              f"RANSAC inliers: {inliers})")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.title(
        f"(d) Final Stitched Panorama\n"
        f"PSNR={metrics['PSNR (dB)']:.2f}dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"RMSE={metrics['RMSE']:.2f}  |  MSE={metrics['MSE']:.2f}  |  UIQI={metrics['UIQI']:.4f}"
    )
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("panorama_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSaved: panorama_result.png")

else:
    print("Not enough matches found to stitch!")