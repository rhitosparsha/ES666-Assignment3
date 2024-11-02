import numpy as np
import cv2
import glob
import os
from numpy.linalg import svd, inv

class PanaromaStitcher():
    def __init__(self):
        pass

    def cylindrical_warp_image(self, img, f):
        h, w = img.shape[:2]
        K = np.array([[f, 0, w // 2], [0, f, h // 2], [0, 0, 1]])  # Intrinsic camera matrix
        cylindrical_img = np.zeros_like(img)
        
        for y in range(h):
            for x in range(w):
                theta = (x - w // 2) / f
                h_val = (y - h // 2) / f
                X = np.array([np.sin(theta), h_val, np.cos(theta)])
                x_prime = K @ X
                x_prime = x_prime / x_prime[2]
                x_prime = x_prime.astype(int)
                if 0 <= x_prime[0] < w and 0 <= x_prime[1] < h:
                    cylindrical_img[y, x] = img[x_prime[1], x_prime[0]]
        return cylindrical_img

    def find_homography(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            u, v = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        
        A = np.array(A)
        U, S, Vt = svd(A)
        H = Vt[-1].reshape(3, 3)

        return H / H[-1, -1]  # Normalize the homography

    def ransac_homography(self, src_pts, dst_pts, thresh=3.0):
        max_inliers = []
        final_H = None
        
        for _ in range(1000):  # Increase RANSAC iterations for better robustness
            idx = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]

            H = self.find_homography(src_sample, dst_sample)

            src_pts_transformed = []
            for pt in src_pts:
                pt_homog = np.array([pt[0], pt[1], 1])
                dst_estimated = np.dot(H, pt_homog.T)
                dst_estimated = dst_estimated / dst_estimated[2]
                src_pts_transformed.append(dst_estimated[:2])

            src_pts_transformed = np.array(src_pts_transformed)
            errors = np.linalg.norm(dst_pts - src_pts_transformed, axis=1)

            # Reduce threshold for more precision
            inliers = np.where(errors < thresh)[0]
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                final_H = H
        
        # After computing the inliers, estimate the scaling factor to refine alignment
        if final_H is not None:
            src_pts_inliers = src_pts[max_inliers]
            dst_pts_inliers = dst_pts[max_inliers]
            src_distances = np.linalg.norm(src_pts_inliers[:-1] - src_pts_inliers[1:], axis=1)
            dst_distances = np.linalg.norm(dst_pts_inliers[:-1] - dst_pts_inliers[1:], axis=1)
            
            # Avoid division by zero
            non_zero_mask = src_distances > 1e-6
            if np.any(non_zero_mask):
                scale_ratio = np.mean(dst_distances[non_zero_mask] / src_distances[non_zero_mask])
            else:
                scale_ratio = 1.0  # Default to no scaling if all distances are too small
            
            # Apply the scaling correction
            if not np.isclose(scale_ratio, 1.0, atol=0.05):  # If the scale ratio differs significantly
                scale_matrix = np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, 1]])
                final_H = scale_matrix @ final_H
        
        return final_H, max_inliers
    
    def stitch_images(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        img2_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        img2_transformed_corners = cv2.perspectiveTransform(img2_corners, H)
        corners = np.concatenate((img2_transformed_corners, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)
        
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel())

        # Ensure xmin, ymin are valid before negative translation
        xmin = max(-1e6, min(xmin, 1e6))  # Clamp translation to avoid overflow
        ymin = max(-1e6, min(ymin, 1e6))
        
        # Adjust the translation homography matrix
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

        result_img = cv2.warpPerspective(img2, translation @ H, (xmax - xmin, ymax - ymin))
        result_img[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

        return result_img
    
    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        homography_matrix_list = []
        stitched_image = None
        f = 1000  # Example focal length for cylindrical warping

        for i in range(len(all_images) - 1):
            img1 = cv2.imread(all_images[i])
            img2 = cv2.imread(all_images[i + 1])

            img1_cyl = self.cylindrical_warp_image(img1, f)
            img2_cyl = self.cylindrical_warp_image(img2, f)

            # Detect SIFT keypoints and descriptors
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1_cyl, None)
            kp2, des2 = sift.detectAndCompute(img2_cyl, None)

            # Match features
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Estimate Homography using RANSAC
            H, inliers = self.ransac_homography(src_pts, dst_pts)
            homography_matrix_list.append(H)

            if stitched_image is None:
                stitched_image = img1_cyl

            stitched_image = self.stitch_images(stitched_image, img2_cyl, H)

        return stitched_image, homography_matrix_list
