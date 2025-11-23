# Chain-of-Knowledge: Dynamic Knowledge Adapting Pipeline

Reproduction of "Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources" (ICLR 2024).

**Paper:** https://proceedings.iclr.cc/paper_files/paper/2024/file/285ba60a67a66d2efeeb7cb25c5067fe-Paper-Conference.pdf

## Overview

This implementation reproduces the Chain-of-Knowledge pipeline, which consists of three main stages:

1. **Reasoning Preparation**: Generate multiple rationales using chain-of-thought prompting
2. **Dynamic Knowledge Adapting**: Retrieve knowledge from external sources and correct rationales
3. **Answer Consolidation**: Generate final answer from corrected rationales

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
```

### 2. Run Tests

**Smoke Test (Quick):**
```bash
source venv/bin/activate
python scripts/run_single_example.py
```

**Full Evaluation:**
```bash
source venv/bin/activate
python scripts/run_evaluation.py
```

### 3. View Results

After evaluation completes:
```bash
source venv/bin/activate
python scripts/visualize_results.py
```

Results are saved to `data/results/`:
- `evaluation_incremental.json` - Latest results
- `evaluation_YYYYMMDD_HHMMSS.json` - Timestamped results
- `results_chart.png` - Visualization
- `results_table.csv` - Table

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
├── data/               # Datasets and results (gitignored)
└── logs/               # Log files (gitignored)
```

## Results

### Implementation Results (Full Pipeline)

| Dataset | Metric | Result | Expected Range | Paper Baseline | Status |
|---------|--------|--------|----------------|----------------|--------|
| FEVER | Accuracy | 30.0% | 20-40% | 63.4% | Below target |
| HotpotQA | Exact Match | 36.0% | 32-38% | 34.1% | Exceeds target |
| MedMCQA | Accuracy | 52.0% | 45-60% | 70.5% | Below target |
| MMLU Physics | Accuracy | 68.0% | 64-70% | 45.5% | Exceeds target |
| MMLU Biology | Accuracy | 66.0% | 62-70% | 83.0% | Below target |

**Summary:**
- Average Score: 50.4%
- Best Performing: MMLU Physics (68.0%, exceeds paper baseline)
- Exceeds Paper Baseline: HotpotQA (36.0% vs 34.1%), MMLU Physics (68.0% vs 45.5%)

**Note:** Results are based on 40-50 samples per dataset with full pipeline execution. FEVER and MedMCQA require additional optimization for better performance. MMLU Physics significantly exceeds the paper baseline.

## Usage

### Basic Usage

```python
from config.settings import config
from src.models.llm_client import LLMFactory
from src.knowledge.wikipedia_retriever import WikipediaRetriever
from src.pipeline.chain_of_knowledge import ChainOfKnowledge

# Initialize LLM client (Llama 3 70B via Together AI)
llm_client = LLMFactory.create_together_client(
    config.TOGETHER_API_KEY,
    config.TOGETHER_MODEL
)

# Initialize knowledge sources
knowledge_sources = {'wikipedia': WikipediaRetriever()}

# Initialize pipeline
cok = ChainOfKnowledge(llm_client, knowledge_sources)

# Run on a question
result = cok.run("What is the capital of France?")
print(result['answer'])
```

## Models Used

- **All Stages**: Llama 3 70B Chat (via Together AI)
- Single model for consistency across all pipeline stages

## Knowledge Sources

- **Wikipedia**: Primary knowledge source for all domains
- **Wikidata SPARQL**: Query generation implemented, execution stubbed

## Configuration

All configuration is in `config/settings.py`. Key parameters:

- `NUM_RATIONALES`: Number of rationales to generate (default: 5)
- `NUM_RATIONALES_FEVER`: Rationales for FEVER (default: 3, saves tokens)
- `CONSENSUS_THRESHOLD`: Threshold for early stopping (default: 0.5, but 0.7 used)
- `REASONING_TEMPERATURE`: Temperature for reasoning generation (default: 0.7)
- `MAX_TOKENS`: Maximum tokens per API call (default: 1024)

## Evaluation

The evaluation framework supports 5 benchmark datasets:

- **FEVER**: Fact verification (40 samples)
- **HotpotQA**: Multi-hop QA (50 samples)
- **MedMCQA**: Medical MCQ (50 samples)
- **MMLU Physics**: Multiple choice (50 samples)
- **MMLU Biology**: Multiple choice (50 samples)

Results are saved to `data/results/` as JSON files.

## Dependencies

- `together`: Together AI API client
- `python-dotenv`: Environment variable management
- `datasets`: HuggingFace datasets library
- `wikipedia`: Wikipedia API wrapper
- `tqdm`: Progress bars

## License

This is a research reproduction project for educational purposes.
