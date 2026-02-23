import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig  # <-- Updated imports here

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models/medlogix-lora-adapter')

# 1. Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "ruslanmv/ai-medical-chatbot"

def format_instruction(example):
    """Formats the dataset into a standard Prompt/Response structure."""
    text = f"<|user|>\nPatient: {example['Patient']}\n<|assistant|>\nDoctor: {example['Doctor']}</s>"
    return {"text": text}

def main():
    print("🚀 Starting Phase 2: LoRA Fine-Tuning for Medical Domain")

    # 2. Load and Prepare Dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train[:200]")
    dataset = dataset.map(format_instruction)
    print(f"Loaded and formatted {len(dataset)} conversations.")

    # 3. Load Tokenizer and Base Model
    print(f"Loading base model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model (HuggingFace automatically maps to MPS on Mac)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto" 
    )

    # 4. Configure LoRA 
    print("Applying LoRA Configuration...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

  # 5. Define Training Arguments (Optimized for Mac Speed)
    training_args = SFTConfig(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=1, # Lower batch size to save memory
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        num_train_epochs=1, 
        logging_steps=5,
        save_steps=25,
        report_to="none",
        dataset_text_field="text", 
        max_length=256,                 # Cut sequence length in half!
        dataloader_pin_memory=False,    # Removes the MPS warning
        gradient_checkpointing=True     # Drastically reduces memory footprint
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Start Training
    print("🧠 Starting Training Loop...")
    trainer.train()

    # 8. Save the Fine-Tuned Adapter Weights
    print(f"✅ Training Complete! Saving LoRA weights to {MODEL_SAVE_PATH}...")
    trainer.model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()