# phase2_finetune/train_med_lora.py

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# Paths & IDs
DATASET_PATH = "data/processed/med_extraction_finetune.jsonl"
OUTPUT_DIR = "./models/lora_med_extraction"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

def train():
    print("1. Loading Tokenizer and 4-Bit Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization saves VRAM, perfect for a 40GB H100 slice
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("2. Configuring LoRA Adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,           # Rank: higher means more capacity
        lora_alpha=32,  # Scaling factor
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("3. Loading Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    print("4. Starting SFT (Supervised Fine-Tuning)...")
    # 1. Use SFTConfig instead of TrainingArguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        # 2. Move these SFT-specific parameters into the config
        dataset_text_field="text",
        max_length=512, 
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # 3. 'tokenizer' is now passed via 'processing_class'
        processing_class=tokenizer, 
    )
    
    trainer.train()
    
    print("5. Saving the Fine-Tuned Model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training complete. Adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()