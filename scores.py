from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import cv2 as cv
from s3 import *
from MMZ_paper import *
from cpbd.compute import *
import os

def compute_correlation_scores():
    mos_path = 'tid2008/mos_with_names.txt'  
    mos_scores = {}

    with open(mos_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            score = float(parts[0])
            mos_scores[filename] = score

    mos_values = []
    sharpness_scores_s3 = []
    sharpness_scores_mmz = []
    sharpness_scores_cpbd = []

    image_folder = 'tid2008/distorted_images/'

    for filename in os.listdir(image_folder):
        if filename.endswith('.bmp') and filename.startswith('I') and '_08_' in filename:
            path = os.path.join(image_folder, filename)

            # Convert to lowercase to match the keys in mos_scores
            filename = filename.lower()

            if filename in mos_scores:
                mos_values.append( mos_scores[filename])
                # Read the image
                image = cv.imread(path, cv.IMREAD_GRAYSCALE)

                # Calculate the sharpness using cpbd
                cpbd_score = compute(image)

                # Calculate sharpness using S3
                s3_detector = S3()
                sharpness_score_s3 = s3_detector.compute_s3(image)

                # Calculate sharpness using MMZ
                bd = BlurDetector()
                _, padded_image = bd.check_image_size(image)
                sharpness_map = 1 - bd.get_blurness(padded_image)
                score_mmz = np.mean(sharpness_map)

                # Add to relevant dictionaries
                sharpness_scores_s3.append(sharpness_score_s3)
                sharpness_scores_mmz.append(score_mmz)
                sharpness_scores_cpbd.append(cpbd_score)
                
    # Compute correlation coefficients for different methods
    plcc_s3, _ = pearsonr(mos_values, sharpness_scores_s3)
    plcc_mmz, _ = pearsonr(mos_values, sharpness_scores_mmz)
    plcc_cpbd, _ = pearsonr(mos_values, sharpness_scores_cpbd)

    srcc_s3, _ = spearmanr(mos_values, sharpness_scores_s3)
    srcc_mmz, _ = spearmanr(mos_values, sharpness_scores_mmz)
    srcc_cpbd, _ = spearmanr(mos_values, sharpness_scores_cpbd)

    # Save results to a CSV file
    results_df = pd.DataFrame({
        'MOS': mos_values,
        'S3_Sharpness': sharpness_scores_s3,
        'MMZ_Sharpness': sharpness_scores_mmz,
        'CPBD_Sharpness': sharpness_scores_cpbd
    })
    results_df.to_csv('sharpness_scores.csv', index=False)

    # Save correlation results to a CSV file
    correlation_results = pd.DataFrame({
        'Method': ['S3', 'MMZ', 'CPBD'],
        'PLCC': [plcc_s3, plcc_mmz, plcc_cpbd],
        'SRCC': [srcc_s3, srcc_mmz, srcc_cpbd]
    })

    correlation_results.to_csv('correlation_results.csv', index=False)