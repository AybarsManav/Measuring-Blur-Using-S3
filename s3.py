import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def display_float_image(image):
    # Normalize the image to the range [0, 255]
    norm_image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    norm_image = np.uint8(norm_image)
    cv.imshow("Normalized Image", norm_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

class S3:

    def compute_s3(self, image):
        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        s1 = self.compute_s1(gray)
        s2 = self.compute_s2(gray)
        # Take the geometric mean of s1 and s2
        s3 = np.sqrt(s1) * np.sqrt(s2)
        # Compute single number s3
        s3_tilde = np.sort(s3, axis=None)[::-1] # Sort the values in s3 descendingly
        s3_tilde = s3_tilde[0:int(len(s3_tilde) * 0.01)] # Take the first 1% of the values
        s3_single = np.mean(s3_tilde) # Compute the mean of the first 1% of the values

        return s1, s2, s3, s3_single
        
    def compute_s1(self, image):
        block_size = 32 # Size of each block
        stride = block_size // 4 # Stride for blocks
        # Pad image with half of block size using reflection
        padded_image = cv.copyMakeBorder(image, block_size // 2, block_size // 2, block_size // 2, block_size // 2, cv.BORDER_REFLECT)
        height, width = padded_image.shape[:2]
        s1 = np.zeros((height, width), dtype=np.float32)
        # counts = np.zeros((height, width), dtype=np.int32)  # To count the number of blocks that covers each pixel
        for row in range(block_size // 2, height - block_size // 2, stride):
            for col in range(block_size // 2, width - block_size // 2, stride):
                # Extract the block
                block = padded_image[row -  block_size // 2:row + block_size // 2, col -  block_size // 2:col + block_size // 2]
                # Check contrast
                if not self.asses_contrast(block):
                    s1[row - (stride//2):row + (stride//2), col -  (stride//2):col + (stride//2)] = 0
                    continue

                mag_F = self.compute_magnitude_spectrum(block)
                radial_frequencies, magnitudes = self.compute_radial_magnitude_spectrum(mag_F)
                log_frequencies = np.log2(radial_frequencies)
                log_magnitudes = np.log2(magnitudes)

                alpha, beta = self.fit_line(log_frequencies, log_magnitudes)
                # Sample line
                x = np.linspace(log_frequencies[0], log_frequencies[-1], 100)
                y = alpha * x + beta

                s1[row -  (stride//2):row + (stride//2), col - (stride//2):col + (stride//2)] = self.sigmoid(-alpha)
                # s1[row -  (block_size // 2):row + (block_size // 2), col -  (block_size // 2):col + (block_size // 2)] = self.sigmoid(-alpha)
                # counts[row -  (block_size // 2):row + (block_size // 2), col -  (block_size // 2):col + (block_size // 2)] += 1

                # Visualize the radial magnitude spectrum
                # plt.plot(log_frequencies, log_magnitudes, label="Original Data")
                # plt.plot(x, y, label="Fitted Line", linestyle="--")
                # plt.xlabel("Log Radial Frequencies")
                # plt.ylabel("Log Magnitudes")
                # plt.title("Radial Magnitude Spectrum with Fitted Line")
                # plt.legend()
                # plt.grid()
                # plt.show()
                # print("Alpha:", -alpha)
                # display_float_image(block)

        # Normalize by the number of blocks that covers each pixel
        # s1 /= np.maximum(counts, 1)

        # Remove padding
        s1 = s1[block_size // 2:height - (block_size // 2), block_size // 2:width - (block_size // 2)] # Remove padding
        # display_float_image(s1)
        return s1
    
    def compute_s2(self, image):
        block_size = 8 # Size of each block
        stride = block_size // 2 # Stride for blocks (4 pixels overlap)
        # Pad image with half of block size using reflection
        padded_image = cv.copyMakeBorder(image, block_size // 2, block_size // 2, block_size // 2, block_size // 2, cv.BORDER_REFLECT)
        height, width = padded_image.shape[:2]
        s2 = np.zeros((height, width), dtype=np.float32)
        # counts = np.zeros((height, width), dtype=np.int32)  # To count the number of blocks that covers each pixel
        for row in range(stride, height - stride, stride):
            for col in range(stride, width - stride, stride):
                # Extract the block
                block = padded_image[row -  stride:row + stride, col -  stride:col + stride]
                # Measure total variation of every 2x2 smaller blocks in block
                sub_block_tv = []
                for i in range(0, block_size-1):
                    for j in range(0, block_size-1):
                        # Get the 2x2 block
                        sub_block = block[i:i+2, j:j+2]
                        # Compute the total variation
                        tv = np.sum(np.abs(sub_block[0, 0] - sub_block[0, 1])) + \
                            np.sum(np.abs(sub_block[0, 1] - sub_block[1, 1])) + \
                            np.sum(np.abs(sub_block[1, 1] - sub_block[1, 0])) + \
                            np.sum(np.abs(sub_block[1, 0] - sub_block[0, 0])) + \
                            np.sum(np.abs(sub_block[0, 0] - sub_block[1, 1])) + \
                            np.sum(np.abs(sub_block[0, 1] - sub_block[1, 0]))
                        sub_block_tv.append(tv / 255 ) # Normalize the sub_block total variation by 255
                # Take maximum and normalize by 4 (since it is the largest TV possible in a 2x2 block)
                max_tv = np.max(sub_block_tv) / 4
                # Set the value in the s2 map
                s2[row -  (stride//2):row + (stride//2), col -  (stride//2):col + (stride//2)] = max_tv
                # s2[row -  stride:row + stride, col -  stride:col + stride] += max_tv
                # counts[row -  stride:row + stride, col -  stride:col + stride] += 1

        # Average the values in s2 by the number of blocks that covers each pixel
        # s2 /= np.maximum(counts, 1)

        # Remove padding
        s2 = s2[( block_size // 2):height - ( block_size // 2), (block_size//2):width - (block_size//2)]  # Remove padding
        # display_float_image(s2)
        return s2
                        
    def asses_contrast(self, image):
        # Transform image intensities according to paper
        b = 0.7656
        k = 0.0364
        gamma = 2.2
        image = (b + k * image) ** (gamma)
        # Set thresholds according to paper
        T1 = 5
        T2 = 2
        # Compute max and min values
        max_val = np.max(image)
        min_val = np.min(image)
        # Compute mean
        mean_val = np.mean(image)
        if max_val - min_val < T1:
            return False
        if mean_val < T2:
            return False
        return True

    def compute_magnitude_spectrum(self, image):
        # Compute the magnitude spectrum of the image
        N = image.shape[0]
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window
        # Compute the 2D FFT
        f = np.fft.fft2(image)
        # Shift the zero frequency component to the center
        f = np.fft.fftshift(f)
        # Compute the magnitude spectrum
        f = np.abs(f)

        # Visualize the magnitude spectrum
        # vis_f = np.log(1 + f)
        # vis_f = cv.normalize(vis_f, None, 0, 255, cv.NORM_MINMAX)
        # vis_f = np.uint8(vis_f)
        # cv.imshow("Magnitude Spectrum", vis_f)
        # cv.waitKey(0)
        return f
    
    def compute_radial_magnitude_spectrum(self, image):
        # Compute the radial magnitude spectrum of the image
        N = image.shape[0]
        z = np.zeros((N // 2, 1), dtype=np.float32)
        for r in range(0, N // 2):
            sum = 0 # Summation of magnitudes in the radius
            for theta in range(360): # consider 1 degree steps
                theta = np.deg2rad(theta)
                # Convert polar coordinates to Cartesian coordinates
                x_float = r * np.cos(theta) + N // 2
                y_float = r * np.sin(theta) + N // 2
                # Get 4 nearest neighbors
                x0 = int(x_float)
                y0 = int(y_float)
                x1 = int(x_float + 1)
                y1 = int(y_float + 1)
                scale_x = x_float - x0
                scale_y = y_float - y0

                # Check if the coordinates are within bounds
                if x1 <= N//2 and y1 <= N//2:
                    # Get the 4 nearest neighbors
                    f00 = image[y0, x0]
                    f01 = image[y0, x1]
                    f10 = image[y1, x0]
                    f11 = image[y1, x1]
                    # Bilinear interpolation x axis
                    f0 = f00 * (1 - scale_x) + f01 * scale_x
                    f1 = f10 * (1 - scale_x) + f11 * scale_x
                    # Bilinear interpolation y axis
                    f = f0 * (1 - scale_y) + f1 * scale_y
                else:
                    f = image[y0, x0]
                sum += f
            # Compute the average
            sum /= 360
            # Store the result in the radial magnitude spectrum
            z[r] = sum

        f = np.linspace(0, 0.5, N // 2)
        f = f[1:-1]
        z = z[1:-1]
        return f, z

    def fit_line(self, x, y):
        # Fit a line to the data
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c
        
    def sigmoid(self, x):
        # Sigmoid function used to transform the slope in the paper
        return 1 - 1 / (1 + np.exp(-3 * (x - 2)))

if __name__ == "__main__":
    s3 = S3()
    image = cv.imread("tid2008/filtered_images/I23_08_1.bmp")

    [s1, s2, s3, s3_metric] = s3.compute_s3(image)
    # Show original image and computed maps in a single figure
    def display_float_image_for_plt(img, cmap='gray'):
        norm_img = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
        return norm_img

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(display_float_image_for_plt(s1), cmap='gray')
    plt.title("S1")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(display_float_image_for_plt(s2), cmap='gray')
    plt.title("S2")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(display_float_image_for_plt(s3), cmap='gray')
    plt.title("S3")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("S3 Metric:", s3_metric)
