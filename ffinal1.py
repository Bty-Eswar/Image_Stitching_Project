import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func


# =============================================================
# UTILITY: Largest interior rectangle crop (no black borders)
# =============================================================
def largest_interior_rectangle(mask):
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)
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
    return best

def crop_to_content(panorama):
    combined_mask = (cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) > 0)
    x, y, w, h = largest_interior_rectangle(combined_mask)
    if w > 0 and h > 0:
        return panorama[y:y+h, x:x+w]
    contours, _ = cv2.findContours(
        combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return panorama[by:by+bh, bx:bx+bw]


# =============================================================
# CLAHE Preprocessing — better contrast for SIFT
# =============================================================
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# =============================================================
# Exposure Compensation — match brightness in overlap
# =============================================================
def exposure_compensate(img1, img2, mask1, mask2):
    overlap = mask1 & mask2
    if not np.any(overlap):
        return img1, img2
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    for c in range(3):
        mean1 = np.mean(img1_f[:, :, c][overlap])
        mean2 = np.mean(img2_f[:, :, c][overlap])
        if mean1 > 0:
            img1_f[:, :, c] *= (mean2 / mean1)
    return np.clip(img1_f, 0, 255).astype(np.uint8), img2


# =============================================================
# BLENDING METHOD 1: Alpha Blending — Paper Equation (6)
# I = α*I1 + (1-α)*I2,  α = 0.5
# =============================================================
def alpha_blend(warped_img1, warped_img2, mask1, mask2):
    alpha   = 0.5
    overlap = mask1 & mask2
    panorama = warped_img1.copy()
    panorama[~mask1] = warped_img2[~mask1]
    panorama[overlap] = (
        warped_img1[overlap] * alpha +
        warped_img2[overlap] * (1 - alpha)
    ).astype(np.uint8)
    return panorama


# =============================================================
# BLENDING METHOD 2: Feathered Blending (smooth gradient)
# =============================================================
def feathered_blend(warped_img1, warped_img2, mask1, mask2):
    overlap = mask1 & mask2
    dist1   = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    dist2   = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)
    total   = dist1 + dist2
    total[total == 0] = 1
    alpha   = dist1 / total
    panorama = np.zeros_like(warped_img1, dtype=np.float32)
    panorama[mask1 & ~mask2] = warped_img1[mask1 & ~mask2]
    panorama[mask2 & ~mask1] = warped_img2[mask2 & ~mask1]
    for c in range(3):
        a = alpha[overlap]
        panorama[overlap, c] = (warped_img1[overlap, c] * a +
                                warped_img2[overlap, c] * (1 - a))
    return panorama.astype(np.uint8)


# =============================================================
# SECTION 5: PERFORMANCE EVALUATION METRICS
# KEY FIX: Metrics computed on OVERLAP REGION ONLY
# (matches paper methodology — compares aligned pixels only)
# =============================================================
def evaluate_metrics_on_overlap(img2_orig, warped_img1, warped_img2,
                                  panorama_blended, H, tx, ty):
    """
    Correct metric computation:
    - Warp img2 into the panorama canvas space
    - Compare only the overlapping pixel region
    - This matches the paper's evaluation methodology
    """
    h2, w2 = img2_orig.shape[:2]
    canvas_h, canvas_w = panorama_blended.shape[:2]

    # Place img2 on canvas (same as during stitching)
    ref_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ref_canvas[ty:h2+ty, tx:w2+tx] = img2_orig

    # Find overlap: pixels where BOTH warped_img1 and warped_img2 have content
    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0
    overlap = mask1 & mask2

    if not np.any(overlap):
        print("Warning: No overlap found for metric computation!")
        return None

    # Extract overlap pixels from blended result and reference
    blend_pixels = panorama_blended[overlap].astype(np.float64)
    ref_pixels   = ref_canvas[overlap].astype(np.float64)

    # MSE & PSNR on overlap
    mse  = np.mean((blend_pixels - ref_pixels) ** 2)
    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
    rmse = np.sqrt(mse)

    # SSIM — use masked 2D patches
    blend_patch = panorama_blended.copy()
    ref_patch   = ref_canvas.copy()
    blend_patch[~overlap] = 0
    ref_patch[~overlap]   = 0

    g_blend = cv2.cvtColor(blend_patch, cv2.COLOR_BGR2GRAY)
    g_ref   = cv2.cvtColor(ref_patch,   cv2.COLOR_BGR2GRAY)
    ssim_val, _ = ssim_func(g_blend, g_ref, full=True)

    # UIQI on overlap
    b = blend_pixels
    r = ref_pixels
    uiqi_ch = []
    for c in range(3):
        x, y_  = b[:, c], r[:, c]
        mu_x, mu_y  = np.mean(x), np.mean(y_)
        sx2, sy2    = np.var(x), np.var(y_)
        sxy         = np.mean((x - mu_x) * (y_ - mu_y))
        denom       = (mu_x**2 + mu_y**2) * (sx2 + sy2)
        uiqi_ch.append((4 * mu_x * mu_y * sxy) / denom if denom != 0 else 1.0)
    uiqi = float(np.mean(uiqi_ch))

    return {"PSNR": psnr, "SSIM": ssim_val, "RMSE": rmse, "MSE": mse, "UIQI": uiqi}


# =============================================================
# SECTION 4.1 — PREPROCESSING: Load & CLAHE
# =============================================================
img1_orig = cv2.imread("dataset/image33.jpg")
img2_orig = cv2.imread("dataset/image44.jpg")

if img1_orig is None or img2_orig is None:
    print("Error: Images not found! Check your paths.")
    exit()

# CLAHE for better feature detection
img1_clahe = apply_clahe(img1_orig)
img2_clahe = apply_clahe(img2_orig)

# Grayscale for SIFT (Paper Section 4.1)
gray1 = cv2.cvtColor(img1_clahe, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_clahe, cv2.COLOR_BGR2GRAY)


# =============================================================
# SECTION 4.1 — FEATURE DETECTION: SIFT
# =============================================================
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=10)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"Keypoints — Image 1: {len(kp1)}, Image 2: {len(kp2)}")


# =============================================================
# SECTION 4.2 — FEATURE MATCHING: BF Matcher + KNN
# threshold = 0.8 (Paper Section 4.2)
# =============================================================
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
print(f"Good matches (threshold=0.8): {len(good_matches)}")

matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, good_matches, None, flags=2)


# =============================================================
# SECTION 4.3 — RANSAC: Homography Estimation
# =============================================================
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0,
                                         maxIters=2000, confidence=0.995)
    inliers = int(ransac_mask.sum())
    print(f"RANSAC inliers: {inliers} / {len(good_matches)}")


    # =============================================================
    # SECTION 4.4 — IMAGE WARPING: Perspective Warping
    # =============================================================
    h1, w1 = img1_orig.shape[:2]
    h2, w2 = img2_orig.shape[:2]

    corners1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    warped_corners = cv2.perspectiveTransform(corners1, H)
    all_corners    = np.concatenate((warped_corners, corners2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    tx, ty = -xmin, -ymin
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    canvas_w, canvas_h = xmax - xmin, ymax - ymin

    # Warp original images (not CLAHE) for natural color output
    warped_img1 = cv2.warpPerspective(img1_orig, T @ H, (canvas_w, canvas_h))
    warped_img2 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_img2[ty:h2+ty, tx:w2+tx] = img2_orig

    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0

    # Exposure compensation
    warped_img1_ec, warped_img2_ec = exposure_compensate(
        warped_img1, warped_img2, mask1, mask2)


    # =============================================================
    # SECTION 4.4 — BLENDING
    # =============================================================
    pano_alpha   = alpha_blend(warped_img1_ec, warped_img2_ec, mask1, mask2)
    pano_feather = feathered_blend(warped_img1_ec, warped_img2_ec, mask1, mask2)

    output_alpha   = crop_to_content(pano_alpha)
    output_feather = crop_to_content(pano_feather)


    # =============================================================
    # SECTION 5 — METRICS (overlap region only — paper-correct)
    # =============================================================
    m_alpha = evaluate_metrics_on_overlap(
        img2_orig, warped_img1_ec, warped_img2_ec, pano_alpha, H, tx, ty)
    m_feather = evaluate_metrics_on_overlap(
        img2_orig, warped_img1_ec, warped_img2_ec, pano_feather, H, tx, ty)

    print("\n" + "="*65)
    print(f"{'METRIC':<10} {'Alpha Blend (Paper)':>24} {'Feathered Blend':>22}")
    print("="*65)
    print(f"{'PSNR(dB)':<10} {m_alpha['PSNR']:>24.2f} {m_feather['PSNR']:>22.2f}  ↑ higher=better")
    print(f"{'SSIM':<10} {m_alpha['SSIM']:>24.4f} {m_feather['SSIM']:>22.4f}  ↑ higher=better")
    print(f"{'RMSE':<10} {m_alpha['RMSE']:>24.2f} {m_feather['RMSE']:>22.2f}  ↓ lower=better")
    print(f"{'MSE':<10} {m_alpha['MSE']:>24.2f} {m_feather['MSE']:>22.2f}  ↓ lower=better")
    print(f"{'UIQI':<10} {m_alpha['UIQI']:>24.4f} {m_feather['UIQI']:>22.4f}  ↑ higher=better")
    print("="*65)


    # =============================================================
    # DISPLAY — Paper Fig. 4 layout
    # =============================================================
    fig = plt.figure(figsize=(20, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB))
    plt.title("(a) Input Image 1", fontsize=11)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB))
    plt.title("(b) Input Image 2", fontsize=11)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"(c) Matching Feature Points\n"
              f"{len(good_matches)} matches  |  RANSAC inliers: {inliers}", fontsize=11)
    plt.axis('off')

    plt.subplot(2, 3, (4, 5))
    plt.imshow(cv2.cvtColor(output_alpha, cv2.COLOR_BGR2RGB))
    plt.title(
        f"(d) Alpha Blending — Paper Equation (6), α=0.5\n"
        f"PSNR={m_alpha['PSNR']:.2f}dB  |  SSIM={m_alpha['SSIM']:.4f}  |  "
        f"RMSE={m_alpha['RMSE']:.2f}  |  MSE={m_alpha['MSE']:.2f}  |  UIQI={m_alpha['UIQI']:.4f}",
        fontsize=10
    )
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(output_feather, cv2.COLOR_BGR2RGB))
    plt.title(
        f"(e) Feathered Blending (Smooth Gradient)\n"
        f"PSNR={m_feather['PSNR']:.2f}dB  |  SSIM={m_feather['SSIM']:.4f}  |  "
        f"RMSE={m_feather['RMSE']:.2f}  |  MSE={m_feather['MSE']:.2f}  |  UIQI={m_feather['UIQI']:.4f}",
        fontsize=10
    )
    plt.axis('off')

    plt.suptitle("Panoramic Image Stitching — Metrics on Overlap Region (Paper-Correct)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("panorama_optimized.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: panorama_optimized.png")

else:
    print("Not enough matches found to stitch!")