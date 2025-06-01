# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from math import atan2, pi
from sys import argv

import numpy as np
from skimage.feature import canny
from .octave import sobel
# If you want to run this script comment the line above and directly use the following line:
# from octave import sobel

import cv2 as cv
import os
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_folder not in os.sys.path:
    os.sys.path.append(project_folder)


# threshold to characterize blocks as edge/non-edge blocks
THRESHOLD = 0.002

# fitting parameter
BETA = 3.6

# block size
BLOCK_HEIGHT, BLOCK_WIDTH = (64, 64)

# just noticeable widths based on the perceptual experiments
WIDTH_JNB = np.concatenate([5*np.ones(51), 3*np.ones(205)])

def compute_cpbd_blur_map_and_score(image):
    """ Compute a blur map and a single score for the whole image """
    blur_map = compute_cpbd_blur_map(image)
    score = compute(image)
    return blur_map, score

def compute_cpbd_blur_map(image, pad_size=32):
    """ According to the paper "S3: A Spectral and Spatial Measure of Local
    Perceived Sharpness in Natural Images" by Cuong T. Vu et al., applying 
    cpbd on overlapping 64x64 patches of the image, a blur map can be generated
    for comparison with S3. """

    block_size = 64
    stride = 8 # 56 pixels overlap according to the paper
    # Pad the image with reflective padding to avoid edge effects
    padded_image = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_REFLECT)
    cv.imshow("Padded Image", padded_image)
    cv.waitKey(0)
    height, width = padded_image.shape
    padded_image = padded_image.astype(np.float64)
    # In this implementation, we compute the edge map using the whole image and then
    # use blocks of the edge maps. This is done because in small images, edge detectors
    # are too sensitive and even smooth patches can be detected as edges.
    canny_edges = canny(padded_image.astype(np.uint8))
    sobel_edges = sobel(padded_image)
    # cv.imshow("sobel_edges", sobel_edges.astype(np.uint8) * 255)
    # cv.imshow("canny_edges", canny_edges.astype(np.uint8) * 255)
    # cv.waitKey(0)
    blur_map = np.zeros((height, width), dtype=np.float64)
    count_map = np.zeros((height, width), dtype=np.int32)
    for row in range(pad_size, height - pad_size, stride):
        for col in range(pad_size, width - pad_size, stride):
            # Get blocks of the image and the edge maps
            block = padded_image[row - pad_size:row + pad_size, col - pad_size:col + pad_size]
            canny_block = canny_edges[row - pad_size:row + pad_size, col - pad_size:col + pad_size]
            sobel_block = sobel_edges[row - pad_size:row + pad_size, col - pad_size:col + pad_size]
            # Compute the CPBD value for the block
            cpbd_value = compute_for_blockwise_cpbd(block, canny_block, sobel_block)
            # Instead of updating only the center, accumulate the CPBD values from surrounding blocks
            # This is done to avoid edge effects and to ensure that the blur map is smooth
            # blur_map[row - (stride//2):row + (stride//2), col - (stride//2):col + (stride//2)] = cpbd_value
            # Update the blur map and count map
            blur_map[row - pad_size:row + pad_size, col - pad_size:col + pad_size] += cpbd_value
            count_map[row - pad_size:row + pad_size, col - pad_size:col + pad_size] += 1
    # Normalize the blur map by the count map to get the average CPBD value
    blur_map /= np.maximum(count_map, 1)  # Avoid division by zero

    # Crop the blur map to the original image size
    return blur_map[pad_size:-pad_size, pad_size:-pad_size]

def compute_for_blockwise_cpbd(block, canny_block, sobel_block):
    # edge width calculation
    marziliano_widths = marziliano_method(sobel_block, block)

    # sharpness metric calculation
    return _calculate_sharpness_metric(block, canny_block, marziliano_widths)

def compute(image):
    # type: (numpy.ndarray) -> float
    """Compute the sharpness metric for the given data."""

    # convert the image to double for further processing
    image = image.astype(np.float64)

    # edge detection using canny and sobel canny edge detection is done to
    # classify the blocks as edge or non-edge blocks and sobel edge
    # detection is done for the purpose of edge width measurement.

    # NOTE: This canny use is wrong. Because for hysteresis, this function uses
    # dtype's maximum value which is incredibly high for float64.
    # We pass as uint8 instead for better results. -Aybars and Jelena
    # canny_edges = canny(image) 
    canny_edges = canny(image.astype(np.uint8)) 
    sobel_edges = sobel(image)

    # edge width calculation
    marziliano_widths = marziliano_method(sobel_edges, image)

    # sharpness metric calculation
    return _calculate_sharpness_metric(image, canny_edges, marziliano_widths)


def marziliano_method(edges, image):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Calculate the widths of the given edges.

    :return: A matrix with the same dimensions as the given image with 0's at
        non-edge locations and edge-widths at the edge locations.
    """

    # `edge_widths` consists of zero and non-zero values. A zero value
    # indicates that there is no edge at that position and a non-zero value
    # indicates that there is an edge at that position and the value itself
    # gives the edge width.
    edge_widths = np.zeros(image.shape)

    # find the gradient for the image
    gradient_y, gradient_x = np.gradient(image)

    # dimensions of the image
    img_height, img_width = image.shape

    # holds the angle information of the edges
    edge_angles = np.zeros(image.shape)

    # calculate the angle of the edges
    for row in range(img_height):
        for col in range(img_width):
            if gradient_x[row, col] != 0:
                edge_angles[row, col] = atan2(gradient_y[row, col], gradient_x[row, col]) * (180 / pi)
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == 0:
                edge_angles[row,col] = 0
            elif gradient_x[row, col] == 0 and gradient_y[row, col] == pi/2:
                edge_angles[row, col] = 90


    if np.any(edge_angles):

        # quantize the angle
        quantized_angles = 45 * np.round(edge_angles / 45)

        for row in range(1, img_height - 1):
            for col in range(1, img_width - 1):
                if edges[row, col] == 1:
                    # Only considers vertical edges, also note that gradient points toward the increasing direction
                    # gradient angle = 180 or -180
                    # The intensity increases towards left during the edge
                    if quantized_angles[row, col] == 180 or quantized_angles[row, col] == -180:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity increasing from left to right
                            # Stop when increase from left to right stops (when left is lower than right which violates the gradient direction)
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) <= 0:
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            #  Stop when increase from left to right stops (when right is higher than left)
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) >= 0:
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_left + width_right


                    # gradient angle = 0
                    if quantized_angles[row, col] == 0:
                        for margin in range(100 + 1):
                            inner_border = (col - 1) - margin
                            outer_border = (col - 2) - margin

                            # outside image or intensity decreasing from left to right
                            if outer_border < 0 or (image[row, outer_border] - image[row, inner_border]) >= 0:
                                break

                        width_left = margin + 1

                        for margin in range(100 + 1):
                            inner_border = (col + 1) + margin
                            outer_border = (col + 2) + margin

                            # outside image or intensity decreasing from left to right
                            if outer_border >= img_width or (image[row, outer_border] - image[row, inner_border]) <= 0:
                                break

                        width_right = margin + 1

                        edge_widths[row, col] = width_right + width_left

    return edge_widths


def _calculate_sharpness_metric(image, edges, edge_widths):
    # type: (numpy.array, numpy.array, numpy.array) -> numpy.float64

    # get the size of image
    img_height, img_width = image.shape

    total_num_edges = 0
    hist_pblur = np.zeros(101)

    # maximum block indices
    num_blocks_vertically = int(img_height / BLOCK_HEIGHT)
    num_blocks_horizontally = int(img_width / BLOCK_WIDTH)

    #  loop over the blocks
    for i in range(num_blocks_vertically):
        for j in range(num_blocks_horizontally):

            # get the row and col indices for the block pixel positions
            rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
            cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))

            if is_edge_block(edges[rows, cols], THRESHOLD):
                block_widths = edge_widths[rows, cols]
                # rotate block to simulate column-major boolean indexing
                block_widths = np.rot90(np.flipud(block_widths), 3)
                block_widths = block_widths[block_widths != 0]

                block_contrast = get_block_contrast(image[rows, cols])
                block_jnb = WIDTH_JNB[block_contrast]

                # calculate the probability of blur detection at the edges
                # detected in the block
                prob_blur_detection = 1 - np.exp(-abs(block_widths/block_jnb) ** BETA)

                # update the statistics using the block information
                for probability in prob_blur_detection:
                    bucket = int(round(probability * 100))
                    hist_pblur[bucket] += 1
                    total_num_edges += 1

    # normalize the pdf
    if total_num_edges > 0:
        hist_pblur = hist_pblur / total_num_edges
    else:
        return 0.0

    # calculate the sharpness metric
    return np.sum(hist_pblur[:64])


def is_edge_block(block, threshold):
    # type: (numpy.ndarray, float) -> bool
    """Decide whether the given block is an edge block."""
    return np.count_nonzero(block) > (block.size * threshold)


def get_block_contrast(block):
    # type: (numpy.ndarray) -> int
    return int(np.max(block) - np.min(block))


if __name__ == '__main__':
    img = cv.imread("tid2008/reference_images/I23.bmp", cv.IMREAD_GRAYSCALE)
    cpbd_blur_map = compute_cpbd_blur_map(img)
    # print("CPBD: ", cpbd)
    cv.imshow("CPBD Blur Map", cv.normalize(cpbd_blur_map, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_64F))
    cv.waitKey(0)