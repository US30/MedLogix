# phase3_rlhf/reward_model.py

from fda_hallucination_check import count_hallucinated_drugs

# Keywords we want to encourage in a Pharmacological Safety Assistant
SAFETY_PHRASES =[
    "consult a physician", 
    "please verify", 
    "interaction database",
    "monitor",
    "clinical judgment",
    "warning",
    "risk"
]

# Keywords we want to penalize (definitive diagnostic language)
DANGER_PHRASES =[
    "you have", 
    "the diagnosis is", 
    "definitely",
    "100% safe",
    "no risk"
]

def score_response(text: str) -> float:
    """
    Calculates a reward score for the PPO trainer based on safety heuristics.
    Target range: -5.0 to +5.0
    """
    score = 0.0
    text_lower = text.lower()
    
    # 1. Reward safety hedging (+1 per phrase, max +3)
    safety_matches = sum(1 for phrase in SAFETY_PHRASES if phrase in text_lower)
    score += min(3.0, safety_matches * 1.0)
    
    # 2. Penalize definitive diagnostic claims (-2 per phrase)
    danger_matches = sum(1 for phrase in DANGER_PHRASES if phrase in text_lower)
    score -= (danger_matches * 2.0)
    
    # 3. HEAVY Penalty for hallucinating fake drugs (-3 per fake drug)
    fake_drugs = count_hallucinated_drugs(text_lower)
    score -= (fake_drugs * 3.0)
    
    # 4. Reward structure (Does it actually provide a Medication Extract?)
    if "medication extract" in text_lower:
        score += 1.0
        
    # Clamp score between -5.0 and +5.0 to keep PPO stable
    return max(-5.0, min(5.0, score))

def get_rewards(decoded_responses: list[str]) -> list[float]:
    """Helper function to score a batch of responses."""
    return [score_response(response) for response in decoded_responses]