from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import cv2 as cv
from s3 import *
from MMZ_paper import *
from cpbd.compute import *
import os
from scipy.optimize import curve_fit

def fit_and_transform(x, y):
    # Define the logistic function
    def f(x, tau1, tau2, tau3, tau4):
        return (tau1 - tau2) / (1 + np.exp((x - tau3) / tau4)) + tau2

    initial_guess = [1, 0.1, 0.1, 1]
    popt, _ = curve_fit(f, x, y, p0=initial_guess, maxfev=10000)
    tau1, tau2, tau3, tau4 = popt
    x_transformed = f(x, tau1, tau2, tau3, tau4)
    return x_transformed

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
                sharpness_score_s3 = s3_detector.compute_s3(image)[3]

                # Calculate sharpness using MMZ
                bd = BlurDetector()
                _, padded_image = bd.check_image_size(image)
                sharpness_map = 1 - bd.get_blurness(padded_image)
                score_mmz = np.mean(sharpness_map)

                # Add to relevant dictionaries
                sharpness_scores_s3.append(sharpness_score_s3)
                sharpness_scores_mmz.append(score_mmz)
                sharpness_scores_cpbd.append(cpbd_score)
                
                
    # Save results to a CSV file
    results_df = pd.DataFrame({
        'MOS': mos_values,
        'S3_Sharpness': sharpness_scores_s3,
        'MMZ_Sharpness': sharpness_scores_mmz,
        'CPBD_Sharpness': sharpness_scores_cpbd
    })
    results_df.to_csv('sharpness_scores.csv', index=False)

    # Fit and transform the MOS values
    s3_scores_transformed = fit_and_transform(sharpness_scores_s3, mos_values)
    mmz_scores_transformed = fit_and_transform(sharpness_scores_mmz, mos_values)
    cpbd_scores_transformed = fit_and_transform(sharpness_scores_cpbd, mos_values)

    # Compute correlation coefficients for different methods
    plcc_s3, _ = pearsonr(mos_values, s3_scores_transformed)
    plcc_mmz, _ = pearsonr(mos_values, mmz_scores_transformed)
    plcc_cpbd, _ = pearsonr(mos_values, cpbd_scores_transformed)

    srcc_s3, _ = spearmanr(mos_values, s3_scores_transformed)
    srcc_mmz, _ = spearmanr(mos_values, mmz_scores_transformed)
    srcc_cpbd, _ = spearmanr(mos_values, cpbd_scores_transformed)

    # Save correlation results to a CSV file
    correlation_results = pd.DataFrame({
        'Method': ['S3', 'MMZ', 'CPBD'],
        'PLCC': [plcc_s3, plcc_mmz, plcc_cpbd],
        'SRCC': [srcc_s3, srcc_mmz, srcc_cpbd]
    })

    correlation_results.to_csv('correlation_results.csv', index=False)