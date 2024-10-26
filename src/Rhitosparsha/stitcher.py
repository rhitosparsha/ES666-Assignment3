import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def normalize_points(self, pts):
        # Normalize the points to improve numerical stability
        mean = np.mean(pts, axis=0)
        std = np.std(pts)
        norm_pts = (pts - mean) / std

        # Create transformation matrix
        T = np.array([
            [1/std, 0, -mean[0]/std],
            [0, 1/std, -mean[1]/std],
            [0, 0, 1]
        ])
        
        return norm_pts, T

    def compute_homography(self, src_pts, dst_pts):
        # Normalize points
        src_pts_norm, T_src = self.normalize_points(src_pts)
        dst_pts_norm, T_dst = self.normalize_points(dst_pts)

        # Build the A matrix for DLT
        A = []
        for i in range(len(src_pts_norm)):
            x, y = src_pts_norm[i][0], src_pts_norm[i][1]
            u, v = dst_pts_norm[i][0], dst_pts_norm[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        
        A = np.array(A)
        
        # Solve the system using SVD
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # Denormalize homography matrix
        H = np.dot(np.linalg.inv(T_dst), np.dot(H, T_src))
        
        # Normalize to make H[2, 2] = 1
        H = H / H[-1, -1]

        return H
    
    def make_panaroma_for_images_in(self,path):
        # Load all images
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')

        # List to store the homography matrices for each image pair
        homography_matrix_list = []

        # Use SIFT to detect features and descriptors
        sift = cv2.SIFT_create()

        # Load the first image and convert to grayscale for feature detection
        base_img = cv2.imread(all_images[0])
        gray_base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        keypoints_base, descriptors_base = sift.detectAndCompute(gray_base_img, None)

        # Iterate through the remaining images and stitch them to the base image
        for i in range(1, len(all_images)):
            next_img = cv2.imread(all_images[i])
            gray_next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
            keypoints_next, descriptors_next = sift.detectAndCompute(gray_next_img, None)

            # Use Brute Force Matcher to find matches
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(descriptors_base, descriptors_next, k=2)

            # Apply ratio test to keep good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Ensure there are enough good matches to compute a homography (minimum 4 points)
            if len(good_matches) > 4:
                # Extract the matching points from both images
                src_pts = np.float32([keypoints_base[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([keypoints_next[m.trainIdx].pt for m in good_matches])

                # Compute the homography using the custom DLT method
                H = self.compute_homography(src_pts, dst_pts)
                homography_matrix_list.append(H)

                # Warp the next image using the computed homography and stitch it with the base image
                height, width = base_img.shape[:2]
                warped_image = cv2.warpPerspective(next_img, H, (width + next_img.shape[1], height))
                
                # Blend the warped image with the base image
                stitched_image = self.blend_images(warped_image, base_img)

                # Update the base image for the next iteration
                base_img = stitched_image
                keypoints_base, descriptors_base = keypoints_next, descriptors_next
            else:
                print(f"Not enough matches found between images {i-1} and {i}.")

        return base_img, homography_matrix_list

    def blend_images(self, warped_img, base_img):
        # Create a mask for the warped image
        height, width = base_img.shape[:2]
        mask_warped = np.sum(warped_img, axis=2) > 0  # Non-zero pixels in warped image
        mask_base = np.sum(base_img, axis=2) > 0      # Non-zero pixels in base image

        blended_image = np.zeros_like(base_img)

        # Blend images using weighted average where both images have pixels
        blended_image[mask_warped] = warped_img[mask_warped]
        blended_image[mask_base] = base_img[mask_base]

        # Combine the overlapping regions
        overlap_mask = mask_warped & mask_base
        if np.any(overlap_mask):
            blended_image[overlap_mask] = (
                warped_img[overlap_mask] * 0.5 + base_img[overlap_mask] * 0.5
            )

        return blended_image.astype(np.uint8)