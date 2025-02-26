#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:43:03 2025

@author: tisl0004
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import json
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from glob import glob

# Setup logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory paths
DATASET_DIR = "./data/test/"
SUBMISSION_DIR = "./submissions/"
REFERENCE_DIR = os.path.join(DATASET_DIR, "3T")  # Ground truth high-field MRI
LOW_FIELD_DIR = os.path.join(DATASET_DIR, "64mT")  # Ultra-low-field MRI

# Allowed modalities
MODALITIES = ["T1", "T2", "FLAIR"]
NUM_TEST_CASES = 15  # Number of test subjects


def load_nifti(file_path):
    """Load a NIfTI image and return as a NumPy array."""
    try:
        img = nib.load(file_path)
        return img.get_fdata()
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def evaluate_submission(submission_path, reference_path):
    """Compute SSIM and PSNR between enhanced and reference MRI images."""
    enhanced_img = load_nifti(submission_path)
    reference_img = load_nifti(reference_path)
    
    if enhanced_img is None or reference_img is None:
        return None, None  # Skip evaluation if loading fails
    
    # Normalize intensity
    enhanced_img = (enhanced_img - np.min(enhanced_img)) / (np.max(enhanced_img) - np.min(enhanced_img))
    reference_img = (reference_img - np.min(reference_img)) / (np.max(reference_img) - np.min(reference_img))
    
    # Compute metrics
    ssim_score = ssim(enhanced_img, reference_img, data_range=1.0)
    psnr_score = psnr(reference_img, enhanced_img, data_range=1.0)
    
    return ssim_score, psnr_score


def evaluate_all_submissions():
    """Evaluate all submissions and rank them."""
    results = []
    
    for team_folder in sorted(os.listdir(SUBMISSION_DIR)):
        team_path = os.path.join(SUBMISSION_DIR, team_folder)
        
        if not os.path.isdir(team_path):
            continue  # Skip non-directory files
        
        team_scores = {"Team": team_folder, "SSIM": [], "PSNR": []}
        
        for subject_id in range(1, NUM_TEST_CASES + 1):
            for modality in MODALITIES:
                submission_file = os.path.join(team_path, f"subject_{subject_id}_enhanced_{modality}.nii.gz")
                reference_file = os.path.join(REFERENCE_DIR, f"subject_{subject_id}_reference_{modality}.nii.gz")
                
                if not os.path.exists(submission_file):
                    logging.warning(f"Missing submission file for {team_folder} - Subject {subject_id} - {modality}")
                    continue
                
                ssim_score, psnr_score = evaluate_submission(submission_file, reference_file)
                
                if ssim_score is not None:
                    team_scores["SSIM"].append(ssim_score)
                    team_scores["PSNR"].append(psnr_score)
                    logging.info(f"{team_folder} - Subject {subject_id} - {modality}: SSIM={ssim_score:.4f}, PSNR={psnr_score:.4f}")
        
        # Average the metrics across all subjects and modalities
        if team_scores["SSIM"]:
            team_scores["SSIM"] = np.mean(team_scores["SSIM"])
            team_scores["PSNR"] = np.mean(team_scores["PSNR"])
            results.append(team_scores)
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="SSIM", ascending=False)  # Rank by SSIM
    
    # Save leaderboard
    df_results.to_csv("leaderboard.csv", index=False)
    logging.info("Leaderboard generated.")
    
    return df_results


if __name__ == "__main__":
    leaderboard = evaluate_all_submissions()
    print(leaderboard)

