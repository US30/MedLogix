import os
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
SFT_ADAPTER_PATH = os.path.join(BASE_DIR, 'models/medlogix-lora-adapter')
ALIGNED_MODEL_PATH = os.path.join(BASE_DIR, 'models/medlogix-rlhf-aligned')

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("⚖️ Starting Phase 3: RLHF (Proximal Policy Optimization)")

   # 1. Setup PPO Configuration (Updated for latest TRL)
    config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=2, # Tiny batch for Mac Memory
        mini_batch_size=2,
        #optimize_device_cache=True,
    )

    # 2. Load the Tokenizer (Use MODEL_NAME directly)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Load the Model with a "Value Head"
    print("Loading Model with Value Head for Reinforcement Learning...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME, # <-- Use MODEL_NAME directly here too
        peft_config=lora_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 4. Initialize the PPO Trainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None, # TRL automatically creates a frozen reference model
        tokenizer=tokenizer,
    )

    # 5. Define Medical Prompts to train on
    medical_prompts = [
        "Patient: I have a severe headache and stiff neck. Doctor:",
        "Patient: My stomach has been hurting after I eat dairy. Doctor:",
        "Patient: I think I broke my arm, it is swelling fast. Doctor:"
    ]

    print("\n🧠 Starting PPO Reinforcement Learning Loop...")
    
    # 6. The RLHF Loop
    for epoch, prompt in enumerate(medical_prompts):
        print(f"\n--- Training Step {epoch + 1}/{len(medical_prompts)} ---")
        
        # Tokenize the prompt
        query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(ppo_trainer.accelerator.device)[0]

        # Generate a response
        generation_kwargs = {
            "max_new_tokens": 50,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        response_tensor = ppo_trainer.generate([query_tensor], return_prompt=False, **generation_kwargs)[0]
        response_text = tokenizer.decode(response_tensor)

        print(f"Model Generated: {response_text.strip()}")

        # 7. The Programmatic Reward Model (Aligning with Human Values)
        # We penalize hallucinations and definitively diagnosing, we reward cautious clinical advice.
        reward = 0.0
        response_lower = response_text.lower()
        
        if "consult a doctor" in response_lower or "physician" in response_lower or "emergency" in response_lower:
            reward += 2.0  # Big reward for safe advice
        if "diagnose" in response_lower or "passage above" in response_lower:
            reward -= 2.0  # Big penalty for hallucinations / playing doctor

        # Fallback reward: slight penalty if it doesn't mention seeing a doctor
        if reward == 0.0:
            reward = -0.5 

        print(f"🎯 Reward Score Assigned: {reward}")
        reward_tensor = torch.tensor([reward], dtype=torch.float16).to(ppo_trainer.accelerator.device)

        # 8. Optimize Policy (PPO Step)
        # The model updates its internal weights to maximize the reward score next time
        train_stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
        print(f"📈 PPO Objective Loss: {train_stats['ppo/loss/total']:.4f}")

    # 9. Save the Aligned Model
    print(f"\n✅ RLHF Complete! Saving aligned model to {ALIGNED_MODEL_PATH}...")
    ppo_trainer.save_pretrained(ALIGNED_MODEL_PATH)

if __name__ == "__main__":
    main()