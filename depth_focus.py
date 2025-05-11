import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def apply_depth_based_gaussian(rgb, depth_map, focused_depth_val):
    """
    Apply a Gaussian filter to the RGB image based on the depth map.
    The Gaussian filter applied to a pixel's neighborhood changes in
    strength according to how close the pixel's depth is to the focused_depth_val.
    """
    rgb = rgb.astype(np.float32)
    rgb_copy = rgb.copy()
    depth_diff = np.abs(depth_map - focused_depth_val)
    sigma_range = 3
    # Normalize depth diff to be in sigma range
    per_pixel_sigmas = (depth_diff / np.max(depth_diff)) * sigma_range
    # Gauss filter sigmas to avoid artifacts
    per_pixel_sigmas = cv.GaussianBlur(per_pixel_sigmas, (11, 11), 0)
    # Loop through each pixel in the RGB image
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            sigma = per_pixel_sigmas[i, j]
            # Create a Gaussian kernel
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            kernel = cv.getGaussianKernel(kernel_size, sigma)
            kernel = kernel @ kernel.T
            half_kernel_size = kernel_size // 2
            # Apply the gaussian filter around the pixel to update the pixel value
            val = np.zeros(3)
            for di in range(-half_kernel_size, half_kernel_size + 1):
                for dj in range(-half_kernel_size, half_kernel_size + 1):
                    if 0 <= i + di < rgb.shape[0] and 0 <= j + dj < rgb.shape[1]:
                        val += kernel[di + half_kernel_size, dj + half_kernel_size] * rgb[i + di, j + dj]
            rgb_copy[i, j] = val
    return rgb_copy.clip(0, 255).astype(np.uint8)

if __name__ == "__main__":
    rgb = cv.imread("rgb.jpg")
    depth = cv.imread("depth.png", cv.IMREAD_GRAYSCALE)

    # Convert depth to float32
    depth = depth.astype(np.float32)
    # Normalize depth to [0, 1]
    # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    # Apply depth-based Gaussian filter
    focused_depth_val = 51 # Example focused depth value
    rgb_defocused = apply_depth_based_gaussian(rgb, depth, focused_depth_val)
    # Display the result

    cv.imshow("rgb", rgb)
    cv.imshow("rgb defocused", rgb_defocused)
    cv.waitKey(0)
    cv.destroyAllWindows()