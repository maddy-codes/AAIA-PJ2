# Task Lifecycle Protocol

This project adheres to a rigorous, state-based workflow for all assigned tasks to ensure technical integrity and thorough verification.

## 1. Task Initialization (`docs/problem/`)
Every new task must begin with the creation of a tracking file: `docs/problem/<task-slug>.md`.
- **Delegation:** Use **`planner`** for high-level roadmap items or **`codebase_investigator`** for deep technical research.
- **Describe:** State the problem or requirement clearly.
- **Context:** Gather and document all relevant codebase context, dependencies, and constraints.
- **Plan:** Outline a step-by-step strategy for implementation.
- **Test Design:** Define specific test scenarios and write the corresponding unit tests *before* implementation (TDD).

## 2. Implementation (`docs/fixed/`)
Once the plan and tests are ready, transition to the implementation phase.
- **Delegation:** Delegate coding tasks to **`implementation-engineer`** for PyTorch/RL logic.
- **Action:** Apply code changes.
- **State Change:** Move the tracking file from `docs/problem/` to `docs/fixed/`.

## 3. Verification & Iteration
Run all unit tests and validation checks.
- **Delegation:** Use **`experiment-analyst`** to interpret performance results or **`generalist`** for broad bug fixing.
- **Failure:** If tests fail or new problems are discovered, move the tracking file back to `docs/problem/` and repeat the Research/Planning/Act cycle.
- **Success:** Only proceed when all test scenarios are passed and all project standards (linting, types) are met.

## 4. Completion (`docs/tested/`)
Upon successful verification of all conditions:
- **State Change:** Move the tracking file from `docs/fixed/` to `docs/tested/`.
- **Final Recap:** The file should serve as a complete record of the problem, the context, the plan, and the verification results.
