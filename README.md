# Chain-of-Knowledge: Dynamic Knowledge Adapting Pipeline

Reproduction of "Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources" (ICLR 2024).

**Paper:** https://proceedings.iclr.cc/paper_files/paper/2024/file/285ba60a67a66d2efeeb7cb25c5067fe-Paper-Conference.pdf

## Overview

This implementation reproduces the Chain-of-Knowledge pipeline, which consists of three main stages:

1. **Reasoning Preparation**: Generate multiple rationales using chain-of-thought prompting
2. **Dynamic Knowledge Adapting**: Retrieve knowledge from external sources and correct rationales
3. **Answer Consolidation**: Generate final answer from corrected rationales

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
LOG_LEVEL=INFO
```

### 3. Run Smoke Test

Test the pipeline on a few examples:

```bash
python scripts/run_single_example.py
```

### 4. Run Full Evaluation

Evaluate on all benchmark datasets:

```bash
python scripts/run_evaluation.py
```

## Repository Structure

```
dl_project/
├── config/              # Configuration settings
├── src/
│   ├── models/         # LLM client implementations
│   ├── knowledge/      # Knowledge source implementations
│   ├── core/           # Pipeline stage implementations
│   ├── pipeline/       # Main orchestrator
│   └── utils/          # Utilities (prompts, logging)
├── evaluation/         # Evaluation framework
├── scripts/            # Execution scripts
├── data/               # Datasets and results
└── logs/               # Log files
```

## Expected Results

Based on the paper baseline (3-shot):

| Dataset | Metric | Paper Result | Target Range |
|---------|--------|--------------|--------------|
| FEVER | Accuracy | 63.4% | 60-65% |
| HotpotQA | Exact Match | 34.1% | 30-35% |
| MedMCQA | Accuracy | 70.5% | 65-70% |
| MMLU Physics | Accuracy | 45.5% | 40-50% |
| MMLU Biology | Accuracy | 83.0% | 80-85% |

## Usage

### Basic Usage

```python
from config.settings import config
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge

# Initialize clients
gemini_client = LLMFactory.create_gemini_client(
    config.GEMINI_API_KEY,
    config.GEMINI_MODEL
)
groq_client = LLMFactory.create_groq_client(
    config.GROQ_API_KEY,
    config.GROQ_MODEL
)

# Initialize knowledge sources
knowledge_sources = {'wikipedia': WikipediaRetriever()}

# Initialize pipeline
cok = ChainOfKnowledge(gemini_client, groq_client, knowledge_sources)

# Run on a question
result = cok.run("What is the capital of France?")
print(result['answer'])
```

## Models Used

- **Reasoning & Consolidation**: Gemini 2.5-Flash (Google)
- **Query Generation**: Llama-3.1-70b-versatile (via Groq API)

## Knowledge Sources

- **Wikipedia**: Primary knowledge source for all domains
- **Wikidata SPARQL**: Query generation implemented, execution stubbed

## Configuration

All configuration is in `config/settings.py`. Key parameters:

- `NUM_RATIONALES`: Number of rationales to generate (default: 5)
- `CONSENSUS_THRESHOLD`: Threshold for early stopping (default: 0.5)
- `REASONING_TEMPERATURE`: Temperature for reasoning generation (default: 0.7)
- `MAX_TOKENS`: Maximum tokens per API call (default: 1024)

## Evaluation

The evaluation framework supports 5 benchmark datasets:

- FEVER (fact verification)
- HotpotQA (multi-hop QA)
- MedMCQA (medical MCQ)
- MMLU Physics (multiple choice)
- MMLU Biology (multiple choice)

Results are saved to `data/results/` as JSON files.

## Dependencies

- `groq`: Groq API client
- `google-generativeai`: Google Gemini API client
- `python-dotenv`: Environment variable management
- `datasets`: HuggingFace datasets library
- `wikipedia`: Wikipedia API wrapper
- `tqdm`: Progress bars

## License

This is a research reproduction project for educational purposes.

