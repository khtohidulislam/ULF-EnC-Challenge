#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:48:53 2025

@author: tisl0004
"""

import numpy as np
import nibabel as nib
import os

# Constants
NUM_TEST_CASES = 15  # Number of test subjects
MODALITIES = ["T1", "T2", "FLAIR"]
DATASET_DIR = "data/test"
SUBMISSION_DIR = "submissions/team1"

# Function to create dummy NIfTI files
def create_dummy_nifti(file_path, shape=(128, 128, 128)):
    """Creates a dummy NIfTI file with random data."""
    data = np.random.rand(*shape)  # Simulated MRI-like data
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, file_path)

# Create necessary directories
os.makedirs(f"{DATASET_DIR}/3T", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/64mT", exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Generate dummy reference (3T) and submission (enhanced) images
for subject_id in range(1, NUM_TEST_CASES + 1):
    for modality in MODALITIES:
        reference_path = f"{DATASET_DIR}/3T/subject_{subject_id}_reference_{modality}.nii.gz"
        submission_path = f"{SUBMISSION_DIR}/subject_{subject_id}_enhanced_{modality}.nii.gz"
        
        create_dummy_nifti(reference_path)
        create_dummy_nifti(submission_path)

print(f"✅ Dummy test dataset created with {NUM_TEST_CASES} subjects and {len(MODALITIES)} modalities!")
