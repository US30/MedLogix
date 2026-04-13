# MedLogix End-to-End Pharmacological Safety Project

This repository contains the full MedLogix capstone pipeline and output artifacts:

- `medlogix/`: complete codebase and data/model artifacts for a 5-phase medical LLM safety workflow
- `Outputs/`: generated output screenshots from project runs

## Repository Layout

```text
.
├── README.md
├── Outputs/
│   ├── Screenshot 2026-04-11 144630.png
│   ├── Screenshot 2026-04-11 150539.png
│   ├── Screenshot 2026-04-11 150730.png
│   ├── Screenshot 2026-04-11 151024.png
│   ├── Screenshot 2026-04-11 151059.png
│   ├── Screenshot 2026-04-11 151310.png
│   └── Screenshot 2026-04-11 151528.png
└── medlogix/
    ├── download_datasets.py
    ├── requirements.txt
    ├── phase1_rag/
    ├── phase2_finetune/
    ├── phase3_rlhf/
    ├── phase4_agent/
    ├── phase5_ui/
    ├── tests/
    ├── data/
    ├── chroma_db/
    └── models/
```

## What This Project Does

MedLogix is built as a progressive, 5-phase pipeline for safer pharmacology assistance:

1. Phase 1 (`phase1_rag`): Build and query a Chroma vector database of pharmacology guidance.
2. Phase 2 (`phase2_finetune`): Generate synthetic extraction data and LoRA fine-tune Llama-3 for medication extraction.
3. Phase 3 (`phase3_rlhf`): Add a safety-oriented reward function and run PPO-style alignment.
4. Phase 4 (`phase4_agent`): Wrap the model in a LangChain ReAct agent with tool calling for interaction checks + guideline lookup.
5. Phase 5 (`phase5_ui`): Expose the assistant through a Streamlit clinical safety dashboard.

## Detailed Component Guide

### 1) Data Ingestion and Storage

- `medlogix/download_datasets.py`
  - Pulls datasets from Hugging Face and KaggleHub.
  - Stores all raw data under `medlogix/data/raw/`.
- Key raw sources already present:
  - FDA approvals dataset
  - Drug-drug interactions dataset
  - WebMD reviews
  - Medical Meadow WikiDoc + MedQA

### 2) RAG Knowledge Layer (Phase 1)

- `medlogix/phase1_rag/embed_webmd_data.py`
  - Loads `medical_meadow_wikidoc.csv`
  - Chunks text with recursive splitting
  - Embeds with `all-MiniLM-L6-v2`
  - Stores vectors in `medlogix/chroma_db/`
- `medlogix/phase1_rag/drug_retriever.py`
  - Queries Chroma collection `pharma_knowledge_base`
  - Returns top-k relevant context chunks

### 3) LoRA Fine-Tuning (Phase 2)

- `medlogix/phase2_finetune/generate_synthetic.py`
  - Generates synthetic clinical-note extraction samples (`jsonl`)
- `medlogix/phase2_finetune/train_med_lora.py`
  - Loads `Meta-Llama-3-8B-Instruct` in 4-bit
  - Applies PEFT LoRA adapters
  - Trains with TRL `SFTTrainer`
  - Saves adapters to `medlogix/models/lora_med_extraction/`
- `medlogix/phase2_finetune/evaluate_extraction.py`
  - Loads base model + LoRA adapters
  - Runs a test extraction prompt

### 4) RLHF Safety Alignment (Phase 3)

- `medlogix/phase3_rlhf/fda_hallucination_check.py`
  - Builds approved-drug token set from FDA data
  - Penalizes generated fake/unknown drug-like names
- `medlogix/phase3_rlhf/reward_model.py`
  - Scores responses with safety heuristics:
    - reward hedging and structured extraction
    - penalize dangerous certainty and hallucinated drugs
- `medlogix/phase3_rlhf/ppo_safety_train.py`
  - Uses TRL PPO over curated prompts
  - Saves aligned model output to `medlogix/models/final_aligned_model/`

### 5) Agent Tooling and Inference (Phase 4)

- `medlogix/phase4_agent/build_interaction_db.py`
  - Builds SQLite interaction DB at `medlogix/data/db/interactions.db`
- `medlogix/phase4_agent/tools.py`
  - `check_drug_interaction`: checks all pairwise combinations in SQLite
  - `search_pharmacology_guidelines`: retrieves RAG snippets from Chroma
- `medlogix/phase4_agent/safety_agent.py` and `safety_agentv2.py`
  - Configure a ReAct-style LangChain agent executor
  - Add custom output parsing to reduce tool-observation hallucinations

### 6) UI Layer (Phase 5)

- `medlogix/phase5_ui/app.py`
  - Streamlit chat app for clinical note prompts
  - Displays safety warnings / pass states
  - Includes operational disclaimer sidebar

## Tests

- `medlogix/tests/` exists for safety test coverage scaffolding.
- Current listed files are placeholders (empty), ready for expansion:
  - `test_hallucinated_drug.py`
  - `test_severe_interaction.py`

## Large Files and Git LFS

This repository includes large artifacts (models and datasets), including files over 100MB. To keep GitHub pushes stable, Git LFS is used for:

- `*.safetensors`
- `*.bin`
- `*.pt`
- `*.sqlite3`
- `*.csv`
- `*.json`

## Environment Setup

From repository root:

```bash
cd medlogix
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Run Order

From inside `medlogix/`:

```bash
# 0) Download datasets (if needed)
python download_datasets.py

# 1) Build vector DB
python phase1_rag/embed_webmd_data.py

# 2) Generate synthetic fine-tune set
python phase2_finetune/generate_synthetic.py

# 3) Train LoRA adapters
python phase2_finetune/train_med_lora.py

# 4) Align model with PPO reward
python phase3_rlhf/ppo_safety_train.py

# 5) Build interaction SQLite DB
python phase4_agent/build_interaction_db.py

# 6) Launch Streamlit app
streamlit run phase5_ui/app.py
```

## Notes and Current Code Assumptions

- Some scripts assume specific relative paths and naming conventions (for example model path aliases in Phase 4).
- If you move folders, update constants in:
  - `phase1_rag/drug_retriever.py`
  - `phase4_agent/build_interaction_db.py`
  - `phase4_agent/safety_agent.py`
  - `phase4_agent/safety_agentv2.py`

## Outputs Folder

`Outputs/` stores screenshot artifacts from project runs and UI execution captures. These are included in the repository as visual output evidence for the capstone pipeline.

## Disclaimer

MedLogix is an AI-assisted safety/documentation workflow and not a substitute for professional diagnosis or licensed clinical decision-making.
