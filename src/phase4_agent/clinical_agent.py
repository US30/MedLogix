import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DB_PATH = os.path.join(BASE_DIR, 'data/mimic_clinical.db')
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("🤖 Booting up the Medical ReAct Agent...")

    # 1. Connect the Agent to the SQLite Database (The "Tool")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    print(f"Connected to database. Available tables: {db.get_usable_table_names()}")

    # 2. Load the LLM (The "Brain")
    print("Loading reasoning engine (LLM)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
   # We remove device_map="auto" and force it to the Mac GPU (mps)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("mps") # <-- Force hardware acceleration
    
    # We wrap the model in a HuggingFace Pipeline so LangChain can talk to it
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False, # <-- FIX: Use Greedy Decoding to avoid MPS math errors!
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Initialize the SQL Agent
    print("Equipping Agent with SQL Tools (ReAct Framework)...")
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        verbose=True, # Setting verbose=True lets us see the internal ReAct loop!
        handle_parsing_errors=True
    )

    print("\n✅ Agent is ready! (Type 'quit' to exit)")
    print("-" * 50)

    # 4. Interactive Agent Loop
    while True:
        user_input = input("\n🩺 Ask about a patient: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Shutting down the Agent. Goodbye!")
            break
            
        print("\n🔍 Agent Thinking Process:")
        try:
            # The invoke() command triggers the ReAct loop
            response = agent_executor.invoke({"input": user_input})
            print(f"\n👨‍⚕️ Agent Final Answer: {response['output']}")
        except Exception as e:
            print(f"\n⚠️ Agent encountered an error: {e}")
            print("Note: Small 1B parameter models sometimes struggle to output perfect ReAct formatting!")

if __name__ == "__main__":
    main()