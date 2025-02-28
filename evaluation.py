import os
import numpy as np
import nibabel as nib
import pandas as pd
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


def calculate_mae(enhanced_img, reference_img):
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(enhanced_img - reference_img))


def calculate_nmse(enhanced_img, reference_img):
    """Compute Normalized Mean Squared Error (NMSE)."""
    mse = np.mean((enhanced_img - reference_img) ** 2)
    norm_factor = np.mean(reference_img ** 2)
    return mse / norm_factor if norm_factor > 0 else np.inf


def evaluate_submission(submission_path, reference_path):
    """Compute SSIM, PSNR, MAE, and NMSE between enhanced and reference MRI images."""
    enhanced_img = load_nifti(submission_path)
    reference_img = load_nifti(reference_path)
    
    if enhanced_img is None or reference_img is None:
        return None, None, None, None  # Skip evaluation if loading fails
    
    # Normalize intensity
    enhanced_img = (enhanced_img - np.min(enhanced_img)) / (np.max(enhanced_img) - np.min(enhanced_img))
    reference_img = (reference_img - np.min(reference_img)) / (np.max(reference_img) - np.min(reference_img))
    
    # Compute metrics
    ssim_score = ssim(enhanced_img, reference_img, data_range=1.0)
    psnr_score = psnr(reference_img, enhanced_img, data_range=1.0)
    mae_score = calculate_mae(enhanced_img, reference_img)
    nmse_score = calculate_nmse(enhanced_img, reference_img)
    
    return ssim_score, psnr_score, mae_score, nmse_score


def evaluate_all_submissions():
    """Evaluate all submissions and rank them."""
    results = []
    
    for team_folder in sorted(os.listdir(SUBMISSION_DIR)):
        team_path = os.path.join(SUBMISSION_DIR, team_folder)
        
        if not os.path.isdir(team_path):
            continue  # Skip non-directory files
        
        team_scores = {"Team": team_folder, "SSIM": [], "PSNR": [], "MAE": [], "NMSE": []}
        
        for subject_id in range(1, NUM_TEST_CASES + 1):
            for modality in MODALITIES:
                submission_file = os.path.join(team_path, f"subject_{subject_id}_enhanced_{modality}.nii.gz")
                reference_file = os.path.join(REFERENCE_DIR, f"subject_{subject_id}_reference_{modality}.nii.gz")
                
                if not os.path.exists(submission_file):
                    logging.warning(f"Missing submission file for {team_folder} - Subject {subject_id} - {modality}")
                    continue
                
                ssim_score, psnr_score, mae_score, nmse_score = evaluate_submission(submission_file, reference_file)
                
                if ssim_score is not None:
                    team_scores["SSIM"].append(ssim_score)
                    team_scores["PSNR"].append(psnr_score)
                    team_scores["MAE"].append(mae_score)
                    team_scores["NMSE"].append(nmse_score)
                    logging.info(f"{team_folder} - Subject {subject_id} - {modality}: SSIM={ssim_score:.4f}, PSNR={psnr_score:.4f}, MAE={mae_score:.4f}, NMSE={nmse_score:.4f}")
        
        # Average the metrics across all subjects and modalities
        if team_scores["SSIM"]:
            team_scores["SSIM"] = np.mean(team_scores["SSIM"])
            team_scores["PSNR"] = np.mean(team_scores["PSNR"])
            team_scores["MAE"] = np.mean(team_scores["MAE"])
            team_scores["NMSE"] = np.mean(team_scores["NMSE"])
            
            # Compute final score
            final_score = (
                0.7 * team_scores["SSIM"] +
                0.1 * team_scores["PSNR"] +
                0.1 * (1 - team_scores["MAE"]) +  # Inverted MAE (lower is better)
                0.1 * (1 - team_scores["NMSE"])   # Inverted NMSE (lower is better)
            )
            
            team_scores["Final Score"] = final_score
            results.append(team_scores)
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Final Score", ascending=False)  # Rank by Final Score
    
    # Save leaderboard
    df_results.to_csv("leaderboard.csv", index=False)
    logging.info("Leaderboard generated.")
    
    return df_results


if __name__ == "__main__":
    leaderboard = evaluate_all_submissions()
    print(leaderboard)