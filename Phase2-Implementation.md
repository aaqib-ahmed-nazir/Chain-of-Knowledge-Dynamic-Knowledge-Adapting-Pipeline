# Phase 2: Quick Wins Implementation - AI Coding Agent Prompt

## Project Context

This is a reproduction of the Chain-of-Knowledge (ICLR 2024) paper with baseline results:
- FEVER: 70.0%
- HotpotQA: 44.0%
- MedMCQA: 30.0%
- MMLU Physics: 34.0%
- MMLU Biology: 44.0%
- Average: 44.4%

**Repository:** https://github.com/aaqib-ahmed-nazir/Chain-of-Knowledge-Dynamic-Knowledge-Adapting-Pipeline

**Goal:** Implement 3 quick wins to improve results to ~50% average (publishable quality).

---

## Task: Implement 3 Quick Wins

You will implement 3 interconnected improvements to the Chain-of-Knowledge pipeline. Each improvement is self-contained but works together for maximum impact.

### Quick Win 1: Adaptive Temperature Tuning (Estimated: 30 minutes)

**File:** `src/core/reasoning.py`

**Objective:** Replace fixed temperature (0.7) with adaptive temperature based on question complexity.

**Implementation Requirements:**

1. Add new method `_estimate_question_complexity()` to `ReasoningPreparation` class:
   - Input: question string
   - Output: float (temperature value)
   - Logic:
     - Simple questions (contains "what is", "who is", "when was", "where is"): return 0.3
     - Moderate questions (contains "how does", "why", "explain", "compare"): return 0.6
     - Complex questions (contains "analyze", "evaluate", "reconcile", "resolve"): return 0.8
     - Default: return 0.7

2. Modify `generate_rationales()` method:
   - Call `_estimate_question_complexity(question)` at the start
   - Store result in variable `adaptive_temp`
   - Pass `adaptive_temp` to all `self.llm_client.call()` invocations instead of `config.REASONING_TEMPERATURE`
   - Log: `f"Generated {self.k} rationales (temp={adaptive_temp:.1f})"`

3. No breaking changes - all other methods remain identical

**Testing:**
- Run smoke test: `python scripts/run_single_example.py`
- Verify different temperatures are used for different questions
- Check logs for temperature values

---

### Quick Win 2: Few-Shot Prompting for FEVER (Estimated: 45 minutes)

**Files to modify:**
1. `src/utils/prompt_templates.py` - Add new template
2. `src/core/reasoning.py` - Modify `generate_rationales()` to use dataset-aware prompts
3. `src/pipeline/chain_of_knowledge.py` - Pass dataset_name through pipeline

**Objective:** Use specialized few-shot examples for FEVER dataset to improve fact verification.

**Implementation Requirements:**

1. In `src/utils/prompt_templates.py`, add new constant after `REASONING_PROMPT_TEMPLATE`:

```python
FEVER_REASONING_PROMPT_TEMPLATE = """You are a fact verification expert. Determine if the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO.

Examples:
1. Claim: "Paris is the capital of France"
   Reasoning: Paris is widely recognized as the capital and largest city of France.
   Verdict: SUPPORTED

2. Claim: "The Earth is flat"
   Reasoning: Modern science has conclusively shown the Earth is spherical.
   Verdict: REFUTED

3. Claim: "The population of Mars is 1 million"
   Reasoning: No evidence suggests Mars has any human population.
   Verdict: NOT ENOUGH INFO

Now analyze this claim:
Claim: {question}

Reasoning:"""
```

2. In `src/core/reasoning.py`, modify class to accept dataset name:
   - Update `__init__()` to accept optional `dataset_name: str = None` parameter
   - Store as `self.dataset_name = dataset_name`

3. Update `generate_rationales()` method signature to accept `dataset_name: str = None`

4. In `generate_rationales()`, add logic:
   ```python
   def generate_rationales(self, question: str, dataset_name: str = None) -> List[str]:
       """Generate k rationales with adaptive temperature and dataset-specific prompts."""
       
       # Determine prompt template
       if dataset_name == 'fever':
           prompt_template = FEVER_REASONING_PROMPT_TEMPLATE
           adaptive_temp = 0.3  # Lower temp for deterministic fact verification
       else:
           prompt_template = REASONING_PROMPT_TEMPLATE
           adaptive_temp = self._estimate_question_complexity(question)
       
       rationales = []
       for i in range(self.k):
           prompt = prompt_template.format(question=question)
           rationale = self.llm_client.call(prompt, temperature=adaptive_temp)
           rationales.append(rationale)
       
       logger.info(f"Generated {self.k} rationales (dataset={dataset_name}, temp={adaptive_temp:.1f})")
       return rationales
   ```

5. In `src/pipeline/chain_of_knowledge.py`:
   - Modify `run()` method signature: `def run(self, question: str, dataset_name: str = None) -> Dict:`
   - Pass `dataset_name` to `self.reasoning.generate_rationales(question, dataset_name)`

6. In `evaluation/evaluator.py`, modify `_run_inference()` or equivalent:
   - When calling `self.cok_model.run(question)`, pass dataset name: `self.cok_model.run(question, dataset_name)`

**Testing:**
- Run on FEVER only: Verify FEVER gets 0.3 temperature and special prompt
- Run on HotpotQA: Verify it gets adaptive temperature and standard prompt
- Check logs for dataset-specific behavior

---

### Quick Win 3: Consensus Validation (Estimated: 40 minutes)

**File:** `src/core/reasoning.py`

**Objective:** Validate consensus answers before returning early stop to prevent false positives.

**Implementation Requirements:**

1. Add new method to `ReasoningPreparation` class:
   ```python
   def validate_consensus_answer(self, question: str, consensus_answer: str) -> bool:
       """Validate if consensus answer is reasonable for the question."""
       validation_prompt = f"""Given this question and answer, determine if the answer is reasonable and relevant to the question.

Question: {question}
Answer: {consensus_answer}

Is this answer reasonable and relevant? Answer with YES or NO only."""
       
       validation = self.llm_client.call(validation_prompt, temperature=0.0)
       is_valid = 'yes' in validation.lower()
       
       if not is_valid:
           logger.info("Consensus answer failed validation - will use full pipeline")
       else:
           logger.info("Consensus answer validated - using early stop")
       
       return is_valid
   ```

2. In `ChainOfKnowledge.run()` method, find the consensus check section and modify:
   ```python
   # Early stopping if consensus (MODIFIED)
   if self.reasoning.has_consensus(answers, config.CONSENSUS_THRESHOLD):
       consensus_answer = max(set(answers), key=answers.count)
       
       # NEW: Validate consensus answer
       if self.reasoning.validate_consensus_answer(question, consensus_answer):
           logger.info("Early stopping: validated consensus reached")
           return {
               "answer": consensus_answer,
               "rationales": rationales,
               "stage": "consensus_validated",
               "confidence": "high"
           }
       else:
           logger.info("Consensus validation failed - proceeding to full pipeline")
   ```

3. Rest of the pipeline logic remains unchanged - it continues if validation fails

**Testing:**
- Run smoke test and observe which answers pass/fail validation
- Run evaluation and check logs for validation outcomes
- Verify some early stops are prevented

---

## Integration Checklist

After implementing all 3 wins, verify:

1. **Quick Win 1 (Adaptive Temperature):**
   - [ ] `_estimate_question_complexity()` method added to ReasoningPreparation
   - [ ] `generate_rationales()` uses adaptive temperature
   - [ ] Logs show different temperatures for different questions
   - [ ] No syntax errors

2. **Quick Win 2 (FEVER Few-Shot):**
   - [ ] `FEVER_REASONING_PROMPT_TEMPLATE` added to prompt_templates.py
   - [ ] `generate_rationales()` accepts dataset_name parameter
   - [ ] FEVER prompts use special template with examples
   - [ ] Other datasets use standard template
   - [ ] `run()` method in ChainOfKnowledge passes dataset_name
   - [ ] Evaluator passes dataset_name to pipeline
   - [ ] Logs show "dataset=fever" for FEVER questions

3. **Quick Win 3 (Consensus Validation):**
   - [ ] `validate_consensus_answer()` method added to ReasoningPreparation
   - [ ] Validation happens before early stop return
   - [ ] Logs show validation results
   - [ ] Failed validations proceed to full pipeline

---

## Expected Results After Implementation

**Before (Baseline):**
```
FEVER:        70.0%
HotpotQA:     44.0%
MedMCQA:      30.0%
MMLU Physics: 34.0%
MMLU Biology: 44.0%
Average:      44.4%
```

**After (With All 3 Wins):**
```
FEVER:        80.0%  (↑ 10%)
HotpotQA:     48.0%  (↑ 4%)
MedMCQA:      36.0%  (↑ 6%)
MMLU Physics: 39.0%  (↑ 5%)
MMLU Biology: 48.0%  (↑ 4%)
Average:      50.2%  (↑ 5.8%)
```

---

## Execution Instructions

1. **Implement in order:**
   - Start with Quick Win 1 (simplest, no dependencies)
   - Then Quick Win 2 (depends on prompt changes)
   - Finally Quick Win 3 (depends on reasoning module)

2. **Test after each win:**
   ```bash
   python scripts/run_single_example.py
   ```

3. **Run full evaluation after all 3:**
   ```bash
   python scripts/run_evaluation.py
   ```

4. **Compare results:**
   - Results saved to `data/results/`
   - Compare new results with baseline

---

## Code Quality Requirements

- Keep code clean and lean - no unnecessary complexity
- All changes are surgical - modify only what's needed
- Maintain existing function signatures where possible
- Add parameters with default values to avoid breaking changes
- Use existing logging patterns
- Follow existing code style (variable naming, formatting, docstrings)

---

## Debugging Notes

**If Quick Win 1 doesn't show different temperatures:**
- Verify `_estimate_question_complexity()` is being called
- Check logs for temperature values
- Ensure question keywords match the complexity_indicators dict

**If Quick Win 2 doesn't improve FEVER:**
- Verify dataset_name is being passed through the pipeline
- Check logs show "dataset=fever" for FEVER questions
- Verify FEVER_REASONING_PROMPT_TEMPLATE is being used (check prompt in logs)
- Ensure temperature is 0.3 for FEVER (deterministic output)

**If Quick Win 3 shows no validation failures:**
- Some answers may be genuinely reasonable - this is OK
- Check logs for validation prompts being sent
- This is still beneficial - prevents obvious false positives

---

## Deliverables

After implementation:

1. All 3 quick wins integrated and working
2. Single test run showing all improvements active
3. Full evaluation run on all 5 datasets
4. Results showing improvement from baseline
5. Clean commit with message: "Phase 2: Implement adaptive temperature, FEVER few-shot, and consensus validation"

---

## Notes for Implementation

- This is straightforward implementation - no algorithmic changes
- All changes are additive - existing code remains intact
- Expected to take 2-3 hours total
- Should show measurable improvements immediately
- Results will be publishable after implementation

