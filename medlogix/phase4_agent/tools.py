# phase4_agent/tools.py

import sqlite3
import itertools
import sys
import os
from langchain.tools import tool

# Make sure it can find the database from the root directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/db/interactions.db")

@tool
def check_drug_interaction(drugs: str) -> str:
    """
    ALWAYS use this tool to check for known interactions.
    Input MUST be a single string with ALL drug names separated by a comma (e.g., "lisinopril, warfarin, aspirin, ibuprofen").
    """
    # Clean the input into a list of lowercase drug names
    drug_list =[d.strip().lower() for d in drugs.split(",") if d.strip()]
    
    if len(drug_list) < 2:
        return "Error: Please provide at least two drug names separated by a comma."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    found_interactions =[]
    
    # Python automatically generates every possible pair (e.g., 4 drugs = 6 checks)
    for drug_a, drug_b in itertools.combinations(drug_list, 2):
        cursor.execute('''
            SELECT severity, description FROM interactions 
            WHERE (drug_a = ? AND drug_b = ?) 
               OR (drug_a = ? AND drug_b = ?)
        ''', (drug_a, drug_b, drug_b, drug_a))
        
        result = cursor.fetchone()
        
        if result:
            severity, description = result
            # Format each warning cleanly
            found_interactions.append(
                f"- {drug_a.capitalize()} + {drug_b.capitalize()} -> [{severity}] {description}"
            )
            
    conn.close()
    
    # Return a compiled report to the AI
    if found_interactions:
        report = "🚨 INTERACTIONS FOUND:\n" + "\n".join(found_interactions)
        return report
    else:
        return f"✅ No known interactions found in the database between: {', '.join([d.capitalize() for d in drug_list])}."

@tool
def search_pharmacology_guidelines(query: str) -> str:
    """
    Use this tool to look up general pharmacology information, drug uses, or side effects.
    Input should be a short search query.
    """
    try:
        # Dynamically add path to import phase1_rag to prevent import errors
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from phase1_rag.drug_retriever import retrieve_drug_context
        
        chunks = retrieve_drug_context(query, top_k=2)
        if chunks:
            return "\n".join(chunks)
        return "No guidelines found for this query."
    except Exception as e:
        return f"Database search error: {str(e)}"

# List of tools to pass to the LangChain agent
AGENT_TOOLS =[check_drug_interaction, search_pharmacology_guidelines]