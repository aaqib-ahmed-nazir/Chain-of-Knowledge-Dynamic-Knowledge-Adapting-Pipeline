REASONING_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the following question step by step.

Question: {question}

Think about what information is needed to answer this question. Break down the problem and provide your reasoning.

Answer:"""

DOMAIN_IDENTIFICATION_PROMPT_TEMPLATE = """Identify the knowledge domains relevant to answer this question.

Available domains: [factual, medical, physics, biology]

Question: {question}

Relevant domains (select from the list):"""

SPARQL_GENERATION_PROMPT_TEMPLATE = """Convert this sentence to a SPARQL query for Wikidata. The query should retrieve relevant entities and facts.

Sentence: {sentence}

SPARQL Query:"""

MEDICAL_EXTRACTION_PROMPT_TEMPLATE = """Extract key medical information from this sentence:

Sentence: {sentence}

Key medical terms and concepts:"""

NL_QUERY_EXTRACTION_PROMPT_TEMPLATE = """Extract the main search query from this sentence:

Sentence: {sentence}

Search query:"""

RATIONALE_CORRECTION_PROMPT_TEMPLATE = """Given the supporting knowledge, correct or improve the following rationale to make it more accurate.

Original Rationale: {original_rationale}

Supporting Knowledge:
{supporting_knowledge}

Corrected Rationale:"""

ANSWER_CONSOLIDATION_PROMPT_TEMPLATE = """Based on the following reasoning steps, provide a concise final answer to the question.

Question: {question}

Reasoning steps:
{reasoning_steps}

IMPORTANT: Provide ONLY the answer, nothing else. 
- For fact verification (SUPPORTS/REFUTES/NOT ENOUGH INFO): Provide only the label (one of: SUPPORTS, REFUTES, or NOT ENOUGH INFO).
- For multiple choice (A, B, C, D): Provide only the letter.
- For factual questions: Provide only the specific fact or entity name.
- Do NOT include explanations, reasoning, or additional text.

Examples:
- If the claim is supported: "SUPPORTS"
- If the claim is refuted: "REFUTES"  
- If there's not enough information: "NOT ENOUGH INFO"
- For multiple choice: "A" or "B" or "C" or "D"
- For factual: "Paris" or "Einstein"

Final Answer:"""

