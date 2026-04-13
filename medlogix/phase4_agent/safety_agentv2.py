# phase4_agent/safety_agent.py

import sys
sys.modules['apex'] = None
sys.modules['flash_attn'] = None
sys.modules['torchao'] = None 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# --- UPDATED IMPORTS ---
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.tools.render import render_text_description
from tools import AGENT_TOOLS

'''# Ensure tools.py is in the same directory or in the python path
try:
    from tools import AGENT_TOOLS
except ImportError:
    # Fallback if running from a different directory
    from phase4_agent.tools import AGENT_TOOLS '''

# Path to your fine-tuned weights
MODEL_PATH = "medlogix_v2/models/final_aligned_model" 

# ==========================================
# 1. CUSTOM PARSER: The Anti-Hallucination Shield
# ==========================================
class MedLogixOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        # If the AI hallucinates the "Observation:", we instantly chop it off.
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        
        # If the AI outputs both an Action and Final Answer, prioritize Action.
        if "Action:" in text and "Final Answer:" in text:
            text = text.split("Final Answer:")[0]
            
        return super().parse(text)

# ==========================================
# 2. AGENT PIPELINE SETUP
# ==========================================
def setup_agent():
    print("Loading AI Model into Agent Pipeline...")
    
    # 2. FIX FOR TOKENIZER ERROR: Load the official Meta tokenizer to bypass local corruption.
    # use_fast=False avoids the 'ModelWrapper' Rust-backend error.
    print("Loading official Llama-3 tokenizer (Bypassing local corruption)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", 
        use_fast=False
    )
    
    # Load your fine-tuned weights
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa"  # <--- ADD THIS LINE
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.01, # Keep extremely low for robotic accuracy
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Force the LLM to stop generating text the moment it asks for an Observation
    llm = llm.bind(stop=["\nObservation:", "Observation:"])
    
    # The Prompt: Forcing literal copying of physiological effects and the disclaimer
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are MedLogix, an advanced Pharmacological Safety Agent for Physicians.
Your task is to analyze patient profiles, extract medications, check for interactions, and output a professional clinical summary.

TOOLS:
{tools}

STRICT REASONING FORMAT:
Question: the user input
Thought: I need to check for interactions.
Action: check_drug_interaction
Action Input: drug1, drug2, drug3
Observation: [Tool Output]
Thought: I have the interaction report. I will now compile the final clinical summary.
Final Answer: [Write the Clinical Output exactly as formatted below]

CRITICAL INSTRUCTIONS:
1. Pass ALL medications into the check_drug_interaction tool at the SAME TIME.
2. NEVER call the tool twice. 
3. NEVER summarize the Observation! You MUST copy the exact physiological effects, adverse risks, and technical descriptions provided by the tool.
4. You MUST include the ⚠️ DISCLAIMER at the very end of your response. Do not stop typing until the disclaimer is fully printed.

=== CLINICAL OUTPUT TEMPLATE ===
🚨 CLINICAL SAFETY ALERT:[COPY THE EXACT TEXT FROM THE OBSERVATION HERE. State the exact adverse effects on the body.]

📋 STRUCTURED MEDICATION EXTRACT:
• [Drug 1 Name]:[Dosage] [Frequency/Route]
• [Drug 2 Name]: [Dosage][Frequency/Route]

⚠️ DISCLAIMER: This analysis is an AI documentation assistant. Please rely on clinical judgment and consult a physician.
================================

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Inject tool names and descriptions
    prompt = prompt.partial(
        tools=render_text_description(AGENT_TOOLS),
        tool_names=", ".join([t.name for t in AGENT_TOOLS]),
    )
    
    # Manually stitch the agent together using our CUSTOM Parser
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | MedLogixOutputParser()
    )
    
    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=AGENT_TOOLS, 
        verbose=True, 
        max_iterations=6, # Prevent infinite loops
        handle_parsing_errors=True
    )
    
    return agent_executor

# --- Test the Agent ---
if __name__ == "__main__":
    agent = setup_agent()
    
    print("\n" + "="*56)
    test_prompt = ("A 35-year-old female takes Levothyroxine 75mcg every morning for hypothyroidism. She is currently experiencing a mild tension headache and wants to take Acetaminophen 500mg. Please extract and verify if there are any interactions.")
    print(f"USER: {test_prompt}\n")
    
    try:
        response = agent.invoke({"input": test_prompt})
        print("\n" + "="*56)
        print("FINAL AGENT OUTPUT:")
        print(response['output'])
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")