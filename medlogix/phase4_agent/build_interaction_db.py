# phase4_agent/build_interaction_db.py

import sqlite3
import pandas as pd
import os

# Paths
CSV_PATH = "data/raw/drug_drug_interactions/drug_interactions.csv" # Adjust filename if Kaggle changes it
DB_PATH = "data/db/interactions.db"

def build_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print("1. Connecting to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("2. Creating interactions table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_a TEXT NOT NULL,
            drug_b TEXT NOT NULL,
            severity TEXT,
            description TEXT
        )
    ''')
    # Create indexes for super-fast lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drug_a ON interactions(drug_a)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drug_b ON interactions(drug_b)')
    
    print("3. Loading Kaggle dataset...")
    try:
        # If the Kaggle dataset is huge, we load it in chunks
        chunksize = 50000
        for chunk in pd.read_csv(CSV_PATH, chunksize=chunksize):
            # Standardize column names to lower case for easy mapping
            chunk.columns =[col.lower() for col in chunk.columns]
            
            # Assuming columns contain 'drug_1', 'drug_2', 'severity', etc.
            # Adjust these column names based on the exact Kaggle CSV format
            records =[]
            for _, row in chunk.iterrows():
                drug_a = str(row.get('drug_1', row.get('drug_a', ''))).lower()
                drug_b = str(row.get('drug_2', row.get('drug_b', ''))).lower()
                severity = str(row.get('severity', 'moderate')).upper()
                desc = str(row.get('description', 'Interaction found.'))
                
                if drug_a and drug_b:
                    records.append((drug_a, drug_b, severity, desc))
            
            cursor.executemany('''
                INSERT INTO interactions (drug_a, drug_b, severity, description)
                VALUES (?, ?, ?, ?)
            ''', records)
            conn.commit()
            
        print("✅ Database successfully built and populated!")
    except FileNotFoundError:
        print(f"❌ CSV not found at {CSV_PATH}. Make sure the Kaggle dataset downloaded correctly.")
        print("Adding dummy data for testing purposes...")
        dummy_data =[
            ("simvastatin", "clarithromycin", "SEVERE", "Clarithromycin increases Simvastatin levels, causing muscle breakdown (rhabdomyolysis)."),
            ("warfarin", "aspirin", "MAJOR", "Concurrent use significantly increases bleeding risk."),
            ("lisinopril", "ibuprofen", "MODERATE", "NSAIDs may decrease the antihypertensive effect of Lisinopril.")
        ]
        cursor.executemany('INSERT INTO interactions (drug_a, drug_b, severity, description) VALUES (?, ?, ?, ?)', dummy_data)
        conn.commit()
        print("✅ Dummy database built for testing.")

    conn.close()

if __name__ == "__main__":
    build_db()