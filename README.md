# Chain-of-Knowledge: Dynamic Knowledge Adapting Pipeline

Reproduction and enhancement of "Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources" (ICLR 2024).

**Paper:** https://proceedings.iclr.cc/paper_files/paper/2024/file/285ba60a67a66d2efeeb7cb25c5067fe-Paper-Conference.pdf

## Overview

This implementation reproduces and extends the Chain-of-Knowledge pipeline with Phase 2 improvements achieving **+8.8% average improvement** over baseline. The pipeline consists of three main stages:

1. **Reasoning Preparation**: Generate multiple rationales with adaptive temperature and dataset-specific prompts
2. **Dynamic Knowledge Adapting**: Retrieve knowledge from multiple sources (Wikipedia, Wikidata, DuckDuckGo) in parallel
3. **Answer Consolidation**: Generate final answer from corrected rationales with consensus validation

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

### Phase 2 Evaluation Results (Latest)

| Dataset | Samples | Our Result | Paper Baseline | vs Paper |
|---------|---------|------------|----------------|----------|
| **FEVER** | 50 | **68.0%** | 63.4% | **+4.6%** ✅ |
| **HotpotQA** | 50 | **54.0%** | 34.1% | **+19.9%** ✅ |
| **MedMCQA** | 50 | 42.0% | 70.5% | -28.5% |
| **MMLU Physics** | 50 | **46.0%** | 45.5% | **+0.5%** ✅ |
| **MMLU Biology** | 50 | 56.0% | 83.0% | -27.0% |

### Performance Summary

- **Average Score:** 53.2% (Paper: 59.3%)
- **Datasets Exceeding Paper:** 3/5 (FEVER, HotpotQA, MMLU Physics)
- **Best Improvement:** HotpotQA (+19.9% vs paper)

### Key Achievements

- **Strong multi-hop reasoning** on HotpotQA (54.0%) - exceeds paper by +19.9%
- **Excellent fact verification** on FEVER (68.0%) - exceeds paper by +4.6%
- **Competitive physics reasoning** on MMLU Physics (46.0%) - exceeds paper by +0.5%

### Improvement from Baseline

| Dataset | Baseline | Phase 2 | Improvement |
|---------|----------|---------|-------------|
| FEVER | 70.0% | 68.0% | -2.0% |
| HotpotQA | 44.0% | 54.0% | **+10.0%** |
| MedMCQA | 30.0% | 42.0% | **+12.0%** |
| MMLU Physics | 34.0% | 46.0% | **+12.0%** |
| MMLU Biology | 44.0% | 56.0% | **+12.0%** |
| **Average** | 44.4% | 53.2% | **+8.8%** |

**Note:** All results use the full Chain-of-Knowledge pipeline with Llama 3 70B via Together AI.

---

## Phase 2 Improvements

### 1. Adaptive Temperature Tuning
Dynamic temperature (0.3-0.8) based on question complexity for optimal reasoning diversity.

### 2. FEVER Few-Shot Prompting
Specialized 3-shot prompt for fact verification with clear SUPPORTS/REFUTES/NOT ENOUGH INFO examples.

### 3. Consensus Validation
LLM validates consensus answers before early stopping to reduce false positives.

### 4. DuckDuckGo Parallel Search
Web search runs in parallel with Wikipedia/Wikidata for broader knowledge coverage.

### 5. Unified Query Execution
Multi-source fusion with relevance scoring for improved knowledge retrieval.

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

- **Wikipedia**: Primary knowledge source for encyclopedic content
- **Wikidata SPARQL**: Structured knowledge for factual queries
- **DuckDuckGo**: Web search for current facts and broader coverage (Phase 2)

## Configuration

All configuration is in `config/settings.py`. Key parameters:

- `NUM_RATIONALES`: Number of rationales to generate (default: 5)
- `NUM_RATIONALES_FEVER`: Rationales for FEVER (default: 3, saves tokens)
- `CONSENSUS_THRESHOLD`: Threshold for early stopping (default: 0.7)
- `REASONING_TEMPERATURE`: Default temperature (default: 0.7, overridden by adaptive)
- `MAX_TOKENS`: Maximum tokens per API call (default: 1024)

## Evaluation

The evaluation framework supports 5 benchmark datasets:

- **FEVER**: Fact verification (50 samples)
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
- `ddgs`: DuckDuckGo search (Phase 2)
- `tqdm`: Progress bars
- `matplotlib`: Results visualization
- `pandas`: Data analysis

## License

This is a research reproduction project for educational purposes.
