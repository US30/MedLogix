import pandas as pd
import sqlite3
import os

# Define relative paths based on our project structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DB_PATH = os.path.join(BASE_DIR, 'data/mimic_clinical.db')

def main():
    print("🏥 Building MIMIC-IV Clinical SQLite Database...")
    
    # Paths to the CSV files you downloaded
    patients_csv = os.path.join(BASE_DIR, 'data/raw/patients.csv')
    labevents_csv = os.path.join(BASE_DIR, 'data/raw/labevents.csv')
    
    # Create a new SQLite connection (this will create the .db file)
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load the Patients data
    print("Loading patients.csv...")
    if os.path.exists(patients_csv):
        df_patients = pd.read_csv(patients_csv)
        # Write the dataframe to a SQL table named 'patients'
        df_patients.to_sql('patients', conn, if_exists='replace', index=False)
        print(f"✅ Inserted {len(df_patients)} rows into 'patients' table.")
    else:
        print(f"❌ Error: Could not find {patients_csv}. Did you put it in the raw folder?")
        return

    # 2. Load the Lab Events data
    print("Loading labevents.csv (this might take a few seconds)...")
    if os.path.exists(labevents_csv):
        # We only take the critical columns to keep the Agent's context window clean
        df_lab = pd.read_csv(labevents_csv, usecols=['subject_id', 'itemid', 'charttime', 'valuenum', 'valueuom', 'flag'])
        
        # Write the dataframe to a SQL table named 'labevents'
        df_lab.to_sql('labevents', conn, if_exists='replace', index=False)
        print(f"✅ Inserted {len(df_lab)} rows into 'labevents' table.")
    else:
        print(f"❌ Error: Could not find {labevents_csv}. Did you put it in the raw folder?")
        return
        
    conn.close()
    print(f"\n🎉 Database successfully built at: {DB_PATH}")
    print("Your Agent now has a real clinical database to query!")

if __name__ == '__main__':
    main()