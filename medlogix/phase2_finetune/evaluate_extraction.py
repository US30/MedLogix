# phase2_finetune/evaluate_extraction.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_ADAPTER_DIR = "./models/lora_med_extraction"

def evaluate():
    print("Loading base model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Merge the LoRA adapter with the base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()
    
    # A highly messy test prompt it has never seen before
    test_note = "Yeah doc, I take that pink pill for my heart, I think it's Carvedilol 25mg every morning. And I chew a 81mg aspirin every day too."
    
    prompt = (
        "<|system|>\nYou are a Pharmacological Safety Assistant. Extract a clean, structured list of medications, dosages, and frequencies from the messy clinical note.\n"
        f"<|user|>\n{test_note}\n"
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\n📝 Generating Extraction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.1, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- RESULT ---")
    # Cut out the prompt part to only show the model's extraction
    print(result.split("<|assistant|>")[-1].strip())
    print("--------------\n✅ Evaluation Complete.")

if __name__ == "__main__":
    evaluate()