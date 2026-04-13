import os
import shutil
import pandas as pd
from datasets import load_dataset
import kagglehub

# Define the target directory based on our project structure
RAW_DATA_DIR = os.path.join("data", "raw")

# Ensure the target directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def save_hf_dataset(repo_id, filename):
    """Downloads a HuggingFace dataset and saves the 'train' split as a CSV."""
    print(f"\n[HuggingFace] Downloading {repo_id}...")
    try:
        ds = load_dataset(repo_id)
        # Convert the 'train' split to a pandas DataFrame and save as CSV
        df = ds['train'].to_pandas()
        save_path = os.path.join(RAW_DATA_DIR, filename)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved to: {save_path}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

def save_kaggle_dataset(repo_id, folder_name):
    """Downloads a Kaggle dataset via kagglehub and moves it to our data folder."""
    print(f"\n[Kaggle] Downloading {repo_id}...")
    try:
        # kagglehub downloads to a hidden cache folder
        cached_path = kagglehub.dataset_download(repo_id)
        target_path = os.path.join(RAW_DATA_DIR, folder_name)
        
        # If the folder already exists, remove it to avoid copy errors
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            
        # Copy from the cache to our project folder
        shutil.copytree(cached_path, target_path)
        print(f"✅ Saved to: {target_path}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    print("🚀 Starting dataset downloads for MedLogix Pharma...\n")

    # --- PHASE 1: Knowledge Base (RAG) ---
    save_hf_dataset(
        repo_id="medalpaca/medical_meadow_wikidoc", 
        filename="medical_meadow_wikidoc.csv"
    )
    save_kaggle_dataset(
        repo_id="rohanharode07/webmd-drug-reviews-dataset", 
        folder_name="webmd_drug_reviews"
    )

    # --- PHASE 2: Fine-Tuning ---
    save_hf_dataset(
        repo_id="medalpaca/medical_meadow_medqa", 
        filename="medical_meadow_medqa.csv"
    )

    # --- PHASE 3: Safety (RLHF) ---
    save_kaggle_dataset(
        repo_id="protobioengineering/united-states-fda-drugs-feb-2024", 
        folder_name="fda_approved_drugs"
    )

    # --- PHASE 4: Agentic Tools ---
    save_kaggle_dataset(
        repo_id="mghobashy/drug-drug-interactions", 
        folder_name="drug_drug_interactions"
    )

    print("\n🎉 All downloads complete! Check your 'data/raw/' directory.")