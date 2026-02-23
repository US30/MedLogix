import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
ADAPTER_PATH = os.path.join(BASE_DIR, 'models/medlogix-lora-adapter')
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("⏳ Loading Base Model and snapping on LoRA Adapter...")
    
    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto" # Automatically uses your Mac's MPS
    )
    
    # 3. Merge the LoRA adapter with the base model!
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("\n✅ MedLogix AI is ready to see patients! (Type 'quit' to exit)")
    print("-" * 50)
    
    # 4. Interactive Chat Loop
    while True:
        patient_input = input("\n🧑‍🦱 Patient: ")
        if patient_input.lower() in ['quit', 'exit', 'q']:
            print("Closing the clinic. Goodbye!")
            break
            
        # We MUST format the prompt exactly how we trained it in finetune.py
        prompt = f"<|user|>\nPatient: {patient_input}\n<|assistant|>\nDoctor:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,      # Don't let it ramble too long
            temperature=0.3,         # Low temperature to keep the model focused/factual
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output and extract just the doctor's reply
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        doctor_reply = full_response.split("Doctor:")[-1].strip()
        
        print(f"👨‍⚕️ Doctor: {doctor_reply}")

if __name__ == "__main__":
    main()