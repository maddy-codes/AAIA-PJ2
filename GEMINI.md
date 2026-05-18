# Project: Advanced AI Applications (PJ2) - CSY3060

## Core Objective
Develop an algorithmic improvement or variation on the baseline Reinforcement Learning (RL) method reproduced in PJ1. This involves a literature review, implementation in PyTorch, and a comparative performance analysis.

## Technical Context
- **Baseline Method:** LSTM-DQN (Narasimhan et al., 2015).
- **Previous Extension (PJ1):** DistilBERT-DQN.
- **Environment:** TextWorldExpress (Coin Collector, Cooking World).
- **Framework:** Python 3, PyTorch.

## Development Standards
- **Task Lifecycle:** All tasks must follow the protocol defined in `.agents/rules/task-lifecycle.md`.
- **Specialized Agents:** Utilize the project-specific sub-agents for specialized tasks:
    - **`planner`**: Project roadmap and task breakdown.
    - **`literature-reviewer`**: Research and synthesis for the literature review.
    - **`methodology-designer`**: Algorithm design and technical rationale.
    - **`implementation-engineer`**: PyTorch implementation and refactoring.
    - **`experiment-analyst`**: Experimental design, execution, and analysis.
    - **`report-writer`**: Drafting and academic refining of the report.
    - **`demo-script-agent`**: Demo video scripting and preparation.
- **Reproducibility:** Fix random seeds for all experimental runs.
- **Documentation:** Maintain detailed records in `docs/` for all experiments and findings.
- **Style:** Adhere to standard Python (PEP 8) and PyTorch idiomatic patterns.

## Submission Requirements
1. **Written Report:** 2000 words (Harvard referencing).
2. **Implementation:** Source code (.py or .ipynb).
3. **Demo Video:** 5-10 minute presentation.
