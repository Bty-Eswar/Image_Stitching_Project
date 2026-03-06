import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("dataset/image2.jpg")
img2 = cv2.imread("dataset/image1.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m,n in matches:
    if m.distance < 0.8*n.distance:
        good_matches.append(m)

print("Good Matches:", len(good_matches))

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# Warp
h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

warped_corners = cv2.perspectiveTransform(corners_img1,H)

all_corners = np.concatenate((warped_corners,corners_img2),axis=0)

[xmin,ymin] = np.int32(all_corners.min(axis=0).ravel())
[xmax,ymax] = np.int32(all_corners.max(axis=0).ravel())

translation = [-xmin,-ymin]

T = np.array([
[1,0,translation[0]],
[0,1,translation[1]],
[0,0,1]
])

warped_img1 = cv2.warpPerspective(img1, T@H, (xmax-xmin, ymax-ymin))

result = warped_img1.copy()

result[
translation[1]:h2+translation[1],
translation[0]:w2+translation[0]
] = img2

# Feather Blending
blend = warped_img1.copy()

rows,cols = blend.shape[:2]

for y in range(rows):
    for x in range(cols):

        if np.any(warped_img1[y,x]) and np.any(result[y,x]):

            alpha = x/cols

            blend[y,x] = (
                (1-alpha)*warped_img1[y,x] +
                alpha*result[y,x]
            )

        elif np.any(result[y,x]):
            blend[y,x] = result[y,x]

blend = blend.astype(np.uint8)

# Crop
gray = cv2.cvtColor(blend, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

coords = cv2.findNonZero(thresh)

x,y,w,h = cv2.boundingRect(coords)

cropped = blend[y:y+h, x:x+w]

plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title("Panorama using Feather Blending")
plt.axis("off")
plt.show()
