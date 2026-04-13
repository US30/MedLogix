# phase2_finetune/generate_synthetic.py

import json
import random
import os

OUTPUT_FILE = "data/processed/med_extraction_finetune.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Fake data building blocks
SYMPTOMS =["a headache", "high blood pressure", "a sinus infection", "back pain", "high cholesterol"]
MESSY_CONTEXTS =[
    "Patient came in complaining of {symptom}. They told me they usually take {drug1} {dose1} {freq1}, and also mentioned taking {drug2} {dose2} {freq2}.",
    "Pt reports taking {drug1} ({dose1}, {freq1}) for {symptom}. Also using {drug2} {dose2} {freq2} over the counter.",
    "Current meds: {drug1} {dose1} {freq1}. Wants something for {symptom}. Currently also on {drug2} {dose2} {freq2}."
]
DRUGS =[
    ("Lisinopril", "10mg", "once daily"),
    ("Atorvastatin", "20mg", "every night"),
    ("Ibuprofen", "400mg", "as needed for pain"),
    ("Amoxicillin", "500mg", "twice a day"),
    ("Metformin", "1000mg", "with meals")
]

SYSTEM_PROMPT = "You are a Pharmacological Safety Assistant. Extract a clean, structured list of medications, dosages, and frequencies from the messy clinical note."

def generate_dataset(num_samples=500):
    print(f"Generating {num_samples} synthetic training samples...")
    records =[]
    for _ in range(num_samples):
        # Pick random combinations
        d1, d2 = random.sample(DRUGS, 2)
        symptom = random.choice(SYMPTOMS)
        messy_text = random.choice(MESSY_CONTEXTS).format(
            symptom=symptom,
            drug1=d1[0], dose1=d1[1], freq1=d1[2],
            drug2=d2[0], dose2=d2[1], freq2=d2[2]
        )
        
        # Format the expected clean output
        clean_extraction = (
            "MEDICATION EXTRACT:\n"
            f"- {d1[0]}: {d1[1]}, {d1[2]}\n"
            f"- {d2[0]}: {d2[1]}, {d2[2]}"
        )
        
        # Combine into causal LM training string
        full_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{messy_text}\n<|assistant|>\n{clean_extraction}<|end_of_text|>"
        
        records.append({"text": full_text})
        
    with open(OUTPUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ Generated dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()