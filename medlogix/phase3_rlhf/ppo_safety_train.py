# phase3_rlhf/ppo_safety_train.py

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from reward_model import get_rewards

# We load the LoRA fine-tuned model from Phase 2
MODEL_PATH = "./models/lora_med_extraction"
OUTPUT_DIR = "./models/final_aligned_model"

def train_ppo():
    print("1. Initializing PPO Configuration...")
    # PPO requires careful tuning. KL penalty keeps it from producing gibberish just to get high scores.
    config = PPOConfig(
        model_name=MODEL_PATH,
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        init_kl_coef=0.2, # Adaptive KL control
        adap_kl_ctrl=True
    )

    print("2. Loading Tokenizer and Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Define the 4-bit config (using bfloat16 for your H100)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # We need a model with a "Value Head" for PPO (it outputs text + a value score)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        # peft_config removed since the model directory already contains it
    )
    
    # PPO needs a frozen reference model to compare against
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )

    print("3. Generating Training Prompts...")
    # In a full run, you would load these from a dataset. Here we use a predefined list of test cases.
    prompts =[
        "Patient takes Lisinopril and needs a painkiller. Generate medication extract.",
        "Patient is on Warfarin. They want to start taking Aspirin every day.",
        "Diagnose this patient: they have a headache and take Amoxicillin.",
        "Write a summary for a patient on Atorvastatin and Clarithromycin."
    ] * 4 # Duplicate to create a small batch of 16

    print("4. Starting PPO Training Loop...")
    for epoch in range(2): # 2 epochs for demonstration
        print(f"\n--- Epoch {epoch+1} ---")
        
        # Format the prompts
        formatted_prompts =[
            f"<|system|>\nYou are a Pharmacological Safety Assistant.\n<|user|>\n{p}\n<|assistant|>\n" 
            for p in prompts
        ]
        
        # Tokenize prompts
        query_tensors =[tokenizer(p, return_tensors="pt")["input_ids"].squeeze().to("cuda") for p in formatted_prompts]
        
        # Process in batches
        for i in range(0, len(query_tensors), config.batch_size):
            batch_queries = query_tensors[i:i+config.batch_size]
            
            # Step A: Generate responses
            # We use do_sample=True to let the model explore different responses
            generation_kwargs = {
                "max_new_tokens": 100,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            response_tensors = ppo_trainer.generate(batch_queries, **generation_kwargs)
            
            # Step B: Decode responses to string
            batch_responses =[tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Extract just the assistant's reply for scoring
            replies = [resp.split("<|assistant|>")[-1] for resp in batch_responses]
            
            # Step C: Score responses using our Reward Model
            rewards = get_rewards(replies)
            reward_tensors = [torch.tensor(r, dtype=torch.float32).to("cuda") for r in rewards]
            
            # Step D: Run PPO step (Update model weights to favor higher rewards)
            stats = ppo_trainer.step(batch_queries, response_tensors, reward_tensors)
            
            # Log average reward
            avg_reward = sum(rewards) / len(rewards)
            print(f"Batch {i//config.batch_size + 1} | Average Reward: {avg_reward:.2f} | KL Divergence: {stats['objective/kl']:.4f}")

    print("5. Saving Aligned Model...")
    ppo_trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ RLHF Training complete! Safe model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_ppo()