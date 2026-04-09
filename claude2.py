import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt


# ─────────────────────────────────────────────────────────
# METRICS (Paper §5 — equations 7 to 14)
# ─────────────────────────────────────────────────────────

def compute_mse(img1, img2):
    """MSE (paper eq. 14): MSE = Σ[I1(m,n)−I2(m,n)]² / (M×N)"""
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    M, N = i1.shape[:2]
    ch = i1.shape[2] if i1.ndim == 3 else 1
    return np.sum((i1 - i2) ** 2) / (M * N * ch)

def compute_psnr(img1, img2):
    """PSNR (paper eq. 7): PSNR = 10·log10(R²/MSE), R=255"""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * log10((255.0 ** 2) / mse)

def compute_rmse(img1, img2):
    """RMSE (paper eq. 9): RMSE = sqrt(Σ(Pi−Oi)²/n)"""
    return sqrt(compute_mse(img1, img2))

def _ssim_channel(ch1, ch2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x, mu_y = np.mean(ch1), np.mean(ch2)
    sx, sy = np.std(ch1), np.std(ch2)
    sxy = np.mean((ch1 - mu_x) * (ch2 - mu_y))
    return ((2*mu_x*mu_y + C1) * (2*sxy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sx**2 + sy**2 + C2))

def compute_ssim(img1, img2):
    """SSIM (paper eq. 8)"""
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    if i1.ndim == 3:
        return np.mean([_ssim_channel(i1[:,:,c], i2[:,:,c]) for c in range(3)])
    return _ssim_channel(i1, i2)

def _uiqi_channel(ch1, ch2):
    mu_x, mu_y = np.mean(ch1), np.mean(ch2)
    sx, sy = np.std(ch1), np.std(ch2)
    sxy = np.mean((ch1 - mu_x) * (ch2 - mu_y))
    denom = (mu_x**2 + mu_y**2) * (sx**2 + sy**2)
    return 0.0 if denom == 0 else (4 * mu_x * mu_y * sxy) / denom

def compute_uiqi(img1, img2):
    """UIQI (paper eq. 10-13)"""
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    if i1.ndim == 3:
        return np.mean([_uiqi_channel(i1[:,:,c], i2[:,:,c]) for c in range(3)])
    return _uiqi_channel(i1, i2)

def evaluate_and_plot(stitched, reference, title="Image Set"):
    """Compute all 5 metrics and plot bar charts (paper Fig. 5 style)"""
    # Both images must be same size — resize reference to match stitched
    h, w = stitched.shape[:2]
    ref = cv2.resize(reference, (w, h))

    psnr = compute_psnr(stitched, ref)
    ssim = compute_ssim(stitched, ref)
    rmse = compute_rmse(stitched, ref)
    mse  = compute_mse(stitched, ref)
    uiqi = compute_uiqi(stitched, ref)

    print("\n" + "="*50)
    print("PERFORMANCE EVALUATION METRICS (Paper §5)")
    print("="*50)
    print(f"  PSNR  (↑ better, 30-50 dB range) : {psnr:.4f} dB")
    print(f"  SSIM  (↑ better, 0-1 range)       : {ssim:.4f}")
    print(f"  RMSE  (↓ better)                  : {rmse:.4f}")
    print(f"  MSE   (↓ better)                  : {mse:.4f}")
    print(f"  UIQI  (↑ better, 0-1 range)       : {uiqi:.4f}")

    # Bar chart — paper Fig. 5 style
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    colors = ['steelblue', 'darkorange', 'tomato', 'purple', 'seagreen']
    labels = ['PSNR\n(dB)', 'SSIM\n(×100)', 'RMSE', 'MSE\n(÷100)', 'UIQI\n(×100)']
    vals   = [psnr, ssim*100, rmse, mse/100, uiqi*100]

    for i, ax in enumerate(axes):
        ax.bar([title], [vals[i]], color=colors[i], width=0.4)
        ax.set_title(labels[i], fontweight='bold')
        ax.set_ylim(0, max(vals[i] * 1.35, 0.1))
        ax.text(0, vals[i] + vals[i]*0.05, f"{vals[i]:.2f}",
                ha='center', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)

    plt.suptitle("Performance Metrics — Paper §5 (Fig. 5 style)",
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig("output_metrics.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("  [Saved] output_metrics.png")

    return {'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse, 'MSE': mse, 'UIQI': uiqi}


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

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


def feathered_blend(img1, img2, mask1, mask2):
    dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)

    total = dist1 + dist2
    total[total == 0] = 1
    alpha = dist1 / total

    panorama = np.zeros_like(img1, dtype=np.float32)
    only1   = mask1 & ~mask2
    only2   = mask2 & ~mask1
    overlap = mask1 & mask2

    panorama[only1] = img1[only1]
    panorama[only2] = img2[only2]

    for c in range(3):
        a = alpha[overlap]
        panorama[overlap, c] = img1[overlap, c] * a + img2[overlap, c] * (1 - a)

    return panorama.astype(np.uint8)


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────

# 1. LOAD IMAGES
img1 = cv2.imread("dataset/image33.jpg")
img2 = cv2.imread("dataset/image44.jpg")

if img1 is None or img2 is None:
    print("Error: Images not found! Check your paths.")
    exit()

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. FEATURE DETECTION — SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"Keypoints — Image 1: {len(kp1)}, Image 2: {len(kp2)}")

# 3. FEATURE MATCHING — BF + Lowe's Ratio Test (threshold=0.8, paper §4.2)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
print(f"Good matches after ratio test: {len(good_matches)}")

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# 4. HOMOGRAPHY & WARPING (RANSAC — paper §4.3)
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers  = int(mask.sum())
    outliers = len(good_matches) - inliers
    print(f"RANSAC — Inliers: {inliers}, Outliers removed: {outliers}")
    print(f"Homography H:\n{H}")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners    = np.concatenate((warped_corners, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0,  1   ]], dtype=np.float64)

    canvas_w, canvas_h = xmax - xmin, ymax - ymin

    warped_img1 = cv2.warpPerspective(img1, T @ H, (canvas_w, canvas_h))

    warped_img2 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_img2[-ymin:h2-ymin, -xmin:w2-xmin] = img2

    # 5. FEATHERED BLENDING
    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0
    panorama = feathered_blend(warped_img1, warped_img2, mask1, mask2)

    # 6. CROP TO LARGEST INTERIOR RECTANGLE
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

    # 7. DISPLAY — paper Fig. 4 style grid
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("a) Input Image 1"); plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("b) Input Image 2"); plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"c) Matching Feature Points ({len(good_matches)})"); plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.title("d) Final Stitched Panorama"); plt.axis('off')

    plt.suptitle("Paper Fig. 4 — Full Pipeline Output", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig("panorama_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: panorama_result.png")

    # ─────────────────────────────────────────────────────
    # 8. METRICS — paper §5 (PSNR, SSIM, RMSE, MSE, UIQI)
    #
    # WHY img2 as reference:
    #   The stitched panorama contains img2's region. We resize
    #   img2 to the panorama size and compare — this measures how
    #   faithfully the stitching preserved the img2 content.
    #   This avoids the black-pixel problem of comparing against
    #   a large empty canvas (which artificially inflated MSE).
    # ─────────────────────────────────────────────────────
    metrics = evaluate_and_plot(final_output, img2, title="Image Set")

    print("\n" + "="*50)
    print("PIPELINE COMPLETE — FINAL SUMMARY")
    print("="*50)
    print(f"  Keypoints (Image 1)   : {len(kp1)}")
    print(f"  Keypoints (Image 2)   : {len(kp2)}")
    print(f"  Matches after KNN     : {len(good_matches)}")
    print(f"  Inliers after RANSAC  : {inliers}")
    print(f"  Outliers removed      : {outliers}")
    print(f"  Final panorama shape  : {final_output.shape}")
    print(f"  PSNR  : {metrics['PSNR']:.4f} dB")
    print(f"  SSIM  : {metrics['SSIM']:.4f}")
    print(f"  RMSE  : {metrics['RMSE']:.4f}")
    print(f"  MSE   : {metrics['MSE']:.4f}")
    print(f"  UIQI  : {metrics['UIQI']:.4f}")

else:
    print("Not enough matches found to stitch!")