# phase3_rlhf/fda_hallucination_check.py

import os
import re
import pandas as pd

# Path to the Kaggle FDA dataset downloaded earlier
# Note: The exact filename might vary slightly depending on the Kaggle update, 
# typically it's named something like 'drugs.csv' or 'products.txt'. 
# Update the filename if necessary.
FDA_DATA_PATH = "data/raw/fda_approved_drugs"

def load_approved_drugs():
    """Loads a set of all FDA approved drug names (lowercase)."""
    approved_drugs = set()
    try:
        # Find the CSV file in the directory
        csv_files =[f for f in os.listdir(FDA_DATA_PATH) if f.endswith('.csv')]
        if not csv_files:
            print("Warning: No CSV found in FDA data path. Using fallback list.")
            return {"aspirin", "lisinopril", "atorvastatin", "ibuprofen", "amoxicillin", "metformin", "clarithromycin"}
            
        df = pd.read_csv(os.path.join(FDA_DATA_PATH, csv_files[0]), low_memory=False)
        
        # Assuming the column with drug names is 'DrugName' or 'ActiveIngredient'
        # We will flatten all string columns just to be safe
        for col in df.columns:
            if df[col].dtype == object:
                for val in df[col].dropna():
                    # Split by spaces or slashes and add to set
                    words = str(val).lower().replace('/', ' ').split()
                    approved_drugs.update(words)
                    
        return approved_drugs
    except Exception as e:
        print(f"Error loading FDA data: {e}. Using fallback list.")
        return {"aspirin", "lisinopril", "atorvastatin", "ibuprofen", "amoxicillin", "metformin"}

# Global set of approved drugs
APPROVED_DRUGS = load_approved_drugs()

def count_hallucinated_drugs(text: str) -> int:
    """
    Extracts drug-like words from text and checks if they exist in the FDA database.
    Returns the count of hallucinated (fake) drugs.
    """
    # Regex to find typical drug suffixes (e.g., -mab, -nib, -vir, -zole, -mycin, -cillin, -statin)
    # This is a targeted approach to catch AI making up fake chemical names.
    drug_pattern = r'\b[a-z]+(?:mab|nib|vir|zole|mycin|cillin|statin|pril|artan|olol)\b'
    
    words = text.lower().split()
    potential_drugs = set(re.findall(drug_pattern, text.lower()))
    
    fake_count = 0
    for drug in potential_drugs:
        if drug not in APPROVED_DRUGS:
            fake_count += 1
            
    return fake_count