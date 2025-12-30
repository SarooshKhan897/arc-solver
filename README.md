# ARC Solver

A modular, high-performance ARC-AGI puzzle solver using multi-model orchestration.

## Architecture

```
arc-solver/
├── src/
│   ├── config.py           # API keys, model configs, settings
│   ├── models.py           # Data models (Task, SolutionCandidate, etc.)
│   │
│   ├── llms/
│   │   └── client.py       # Shared AsyncOpenAI client with retry logic
│   │
│   ├── perception/
│   │   ├── objects.py      # Object extraction (ObjectPreprocessor)
│   │   ├── perceiver.py    # perceive(grid) → structured analysis
│   │   └── differencer.py  # difference(in, out) → transformation delta
│   │
│   ├── solve/
│   │   ├── executor.py     # execute_transform(code, grid)
│   │   ├── prompt.py       # generate_prompt() for solver
│   │   └── solver.py       # solve_single(), solve_with_models()
│   │
│   ├── verification/
│   │   ├── verifier.py     # verify() → score & verdict
│   │   └── self_verifier.py # self_verify() → CORRECT/WRONG/UNSURE
│   │
│   ├── utils/
│   │   ├── grid.py         # Grid utilities (grid_to_text, diff)
│   │   └── trace.py        # TraceLogger for debugging
│   │
│   ├── run.py              # Task orchestration (solve_task, run_tasks)
│   └── main.py             # CLI entry point
```

## Flow

```
┌─────────────────────────────────────────────────────────┐
│  For each task:                                          │
│                                                          │
│  1. perceiver.perceive()  →  Grid analysis              │
│  2. differencer.difference() → Transformation delta      │
│  3. solver.solve_with_models() → Parallel model calls   │
│         ├── solver.solve_single() for each model        │
│         ├── verifier.verify() → score                   │
│         └── self_verifier.self_verify() → validation    │
│                                                          │
│  Stop when 2 solutions from different models pass       │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone and enter directory
cd arc-solver

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment file
cp env.example .env
# Edit .env with your API key
```

## Usage

### CLI

```bash
# Solve a single task
python -m src.main solve ARC-AGI-2/data/evaluation/abc123.json

# Run 10 random tasks
python -m src.main random -n 10

# Run full evaluation
python -m src.main eval -l 50 -o results.json
```

### Python API

```python
from src.run import solve_task, run_tasks
from src.main import run_single_task_sync
import asyncio

# Single task
result = run_single_task_sync("path/to/task.json")
print(f"Correct: {result['exact_match']}")

# Async usage
async def main():
    # Solve a task
    task_data = {"train": [...], "test": [...]}
    predictions, info = await solve_task(task_data)
    
    # Run multiple tasks
    results = await run_tasks(["task1.json", "task2.json"])

asyncio.run(main())
```

### Using Individual Components

```python
import numpy as np
from src.perception import perceive, difference
from src.solve import solve_single
from src.verification import verify, self_verify
from src.config import SOLVER_MODELS

# Perceive a grid
grid = np.array([[0, 1], [1, 0]])
perception = await perceive(grid)

# Compare grids
delta = await difference(input_grid, output_grid)

# Solve with a specific model
candidate = await solve_single(
    task_data=task,
    model_config=SOLVER_MODELS[0],  # GPT-5.2
    perceptions=perceptions,
    deltas=deltas,
)

# Verify independently
score = await verify(code, explanation, examples, test_input)
decision = await self_verify(model, model_id, ...)
```

## Configuration

Edit `src/config.py` or use environment variables:

```python
# Models
SOLVER_MODELS = [
    {"id": "gpt-5.2", "model": "openai/gpt-5.2", ...},
    {"id": "gemini-pro", "model": "google/gemini-3-pro-preview", ...},
]

# Thresholds
MIN_VERIFIER_SCORE = 90
MIN_SOLUTIONS_REQUIRED = 2

# Concurrency
MAX_WORKERS = 100
```

## Key Features

- **Parallel Execution**: Multiple models run simultaneously
- **Non-blocking**: Tasks don't block each other
- **Modular Functions**: Each component is independently testable
- **Retry with Feedback**: Failed attempts get detailed feedback
- **Self-Verification**: Models verify their own outputs
- **Score Thresholds**: Only accepts high-confidence solutions

## Models Used

- **Perceiver**: Gemini 3 Pro (grid analysis)
- **Differencer**: Claude Opus 4.5 (transformation comparison)
- **Solver**: GPT-5.2, Gemini 3 Pro, Gemini 3 Flash (parallel)
- **Verifier**: GPT-5.2 (high reasoning)
- **Self-Verifier**: Same model that generated the solution

## License

MIT

