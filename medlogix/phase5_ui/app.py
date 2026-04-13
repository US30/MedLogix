# phase5_ui/app.py

import streamlit as st
import sys
import os

# Add the root directory to the path so we can import our Phase 4 Agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase4_agent.safety_agent import setup_agent

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MedLogix | Pharmacological Safety Dashboard",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for a clean medical look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    </style>
    """, unsafe_allow_stdio=True)

# ==========================================
# 2. MODEL CACHING
# ==========================================
@st.cache_resource
def load_medlogix_agent():
    """Loads the Phase 4 Agent once and keeps it in VRAM."""
    return setup_agent()

# ==========================================
# 3. SIDEBAR (Information & Disclaimer)
# ==========================================
with st.sidebar:
    st.title("💊 MedLogix Pharma")
    st.subheader("Pharmacological Safety Assistant")
    st.divider()
    
    st.info("""
    **Core Capabilities:**
    - 🔍 **Drug Interaction Check:** Uses a live SQLite database.
    - 📝 **Medication Extraction:** Structured lists from messy notes.
    - 📚 **Pharmacology RAG:** Access to clinical guidelines.
    """)
    
    st.warning("""
    ⚠️ **OFFICIAL DISCLAIMER:**
    This tool is a documentation and reasoning assistant only. It is NOT a diagnostic tool. 
    Always verify drug interactions with official pharmacy databases.
    """)
    
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 4. MAIN CHAT INTERFACE
# ==========================================
st.title("Prescription Safety Dashboard")
st.caption("Enter patient medications or a clinical note to check for safety risks.")

# Initialize the agent
with st.spinner("Initializing MedLogix AI Model (this may take a minute)..."):
    agent_executor = load_medlogix_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ex: Patient takes Simvastatin 20mg and wants to start Clarithromycin..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Placeholder for the "Thinking" logs
        thought_container = st.expander("🔍 View AI Reasoning Path (ReAct Loop)", expanded=True)
        
        with st.spinner("Analyzing pharmacological safety..."):
            try:
                # We call the agent we built in Phase 4
                # The 'verbose=True' in Phase 4 will still print to your terminal/logs,
                # but we capture the final output here.
                response = agent_executor.invoke({"input": prompt})
                final_answer = response["output"]
                
                # Check if it's a safety warning to color it differently
                if "🚨 SAFETY WARNING" in final_answer:
                    st.error(final_answer)
                elif "✅ SAFETY CHECK PASSED" in final_answer:
                    st.success(final_answer)
                else:
                    st.markdown(final_answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Error: Could not process request."})

# ==========================================
# 5. FOOTER
# ==========================================
st.divider()
st.caption("MedLogix v1.0 | AI/ML Engineering Capstone Project | Built on Llama-3-8B")