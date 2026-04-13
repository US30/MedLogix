# phase4_agent/safety_agent.py

import sys
sys.modules['apex'] = None

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

# Load the final safe model from Phase 3 (or Phase 2 if Phase 3 was skipped)
MODEL_PATH = "medlogix_v2/models/final_aligned_model" 
# Fallback for testing:
# MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

# ==========================================
# 1. CUSTOM PARSER: The Anti-Hallucination Shield
# ==========================================
class MedLogixOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        # If the AI hallucinates the "Observation:", we instantly chop it off.
        # This discards the fake result so LangChain can run your REAL SQLite database tool.
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        
        # If the AI gets confused and outputs both an Action and Final Answer,
        # we force it to prioritize the Action first.
        if "Action:" in text and "Final Answer:" in text:
            text = text.split("Final Answer:")[0]
            
        # Pass the cleaned text back to standard LangChain
        return super().parse(text)

# ==========================================
# 2. AGENT PIPELINE SETUP
# ==========================================
def setup_agent():
    print("Loading AI Model into Agent Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.01, # Keep extremely low for robotic accuracy
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # We still tell LangChain to stop, just in case
    llm = llm.bind(stop=["\nObservation:", "Observation:"])
    
    # The Prompt: Updated to give the AI an escape hatch to the Final Answer
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are MedLogix, a strict Pharmacological Safety Assistant.
Your job is to check drug interactions using the provided tools, and then list the medications safely in your Final Answer. 
DO NOT use tools to extract, clean, or format text. You do the formatting yourself in the Final Answer.

You have access to the following tools:
{tools}

Use the following format strictly:
Question: the input question or patient note
Thought: think about what to do next
Action: the action to take, MUST be exactly one of [{tool_names}]
Action Input: the exact input to the action
Observation: the result of the action
... (Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (or no more tools are needed)
Final Answer: YOUR FINAL RESPONSE.

CRITICAL RULES:
1. NEVER invent tool names. You MUST ONLY use: {tool_names}.
2. NEVER output "Action: None". If you do not need any more tools, immediately output "Final Answer:".
3. NEVER generate the 'Observation:' yourself! After you output 'Action Input:[data]', you MUST STOP typing immediately.
4. If ANY tool found an interaction, your Final Answer MUST start with "🚨 SAFETY WARNING:" and you MUST describe EVERY interaction found (e.g., mention Bleeding Risk and Rhabdomyolysis).
5. If NO interaction was found, your Final Answer MUST start with "✅ SAFETY CHECK PASSED:".
6. You MUST list the exact medications, dosages, and frequencies in a clean bulleted list.
7. You MUST end your Final Answer with the exact sentence: "Please consult a physician for clinical decision-making."

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
        max_iterations=4, # Stop runaway loops
        handle_parsing_errors=True
    )
    
    return agent_executor

# --- Test the Agent ---
if __name__ == "__main__":
    agent = setup_agent()
    
    print("\n========================================================")
    test_prompt = "My 72-year-old patient is currently taking Simvastatin 40mg at night for high cholesterol and Warfarin 5mg for a history of blood clots. They have a severe sinus infection and joint pain. I want to add Clarithromycin 500mg twice daily for the infection and Aspirin 81mg daily for the pain. Can you extract these medications and check if this complete combination is safe?"
    print(f"USER: {test_prompt}\n")
    
    response = agent.invoke({"input": test_prompt})
    
    print("\n========================================================")
    print("FINAL AGENT OUTPUT:")
    print(response['output'])