"""Self-Verifier - model verifies its own output."""

import re
from typing import Any

import numpy as np

from src.llms.client import call_llm
from src.utils.grid import grid_to_text

# =============================================================================
# Self-Verification Prompt
# =============================================================================

SELF_VERIFICATION_PROMPT = """You wrote code to solve an ARC puzzle. I ran your code on ALL {num_tests} test input(s).

## Your Understanding of the Pattern
{explanation}

## Your Code
```python
{code}
```

## Training Examples (ALL) - The Ground Truth
These are the CORRECT input→output transformations that define the pattern:
{training_pairs}

## Test Results ({num_tests} test input(s)):
Your code produced these outputs. Compare them against the training pattern:
{test_results}

## VERIFICATION TASK

**CRITICAL**: Compare your test outputs against the training examples. The transformation applied to test inputs should follow the EXACT same pattern as training.

1. **TRANSFORMATION CHECK**: 
   - Look at Training Input→Output. What transformation happened?
   - Look at Test Input→Your Output. Did the SAME transformation happen?
   - Does your code logic actually implement this transformation?

2. **SHAPE**: Training outputs had shapes: {training_output_shapes}. Do your output shapes follow the same pattern relationship with inputs?

3. **COLORS**: Training outputs used colors: {training_output_colors}. Do your outputs use the expected colors based on the pattern?

4. **VISUAL PATTERN**: Compare side-by-side:
   - Training: Input has X → Output has Y
   - Test: Input has X' → Your output has Y'?
   - Is the relationship consistent?

5. **COMMON ERROR CHECK**:
   - Off-by-one error in positions?
   - Missing or extra objects?
   - Wrong rotation/reflection direction?
   - Foreground/background swapped?
   - Index out of bounds issues?

6. **CODE vs OUTPUT**: Does the code logic produce what you see in the output? Any bugs?

## YOUR RESPONSE (ALL THREE REQUIRED)

**VERDICT** (must be one of these three):
- CORRECT: ALL outputs match the pattern from training examples. The transformation is consistent.
- WRONG: One or more outputs do NOT match the expected pattern.
- UNSURE: Something looks off but I'm not certain.

**SCORE** (0-100):
- 90-100: Highly confident the outputs are correct
- 70-89: Likely correct but minor concerns
- 50-69: Uncertain, could go either way
- 30-49: Likely has issues
- 0-29: Almost certainly wrong

**FEEDBACK**: Brief explanation of your assessment. If WRONG or UNSURE, explain what's wrong and what the output SHOULD look like.

Format your response as:
VERDICT: [CORRECT/WRONG/UNSURE]
SCORE: [0-100]
FEEDBACK: [Your explanation]"""


# =============================================================================
# Self-Verify Function
# =============================================================================

async def self_verify(
    model: str,
    model_id: str,
    extra_body: dict[str, Any] | None,
    max_tokens: int | None,
    code: str,
    explanation: str,
    train_examples: list[dict[str, Any]],
    test_inputs: list[np.ndarray],
    test_outputs: list[np.ndarray],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Have the same model verify its own output(s).

    Args:
        model: Model identifier
        model_id: Short model ID
        extra_body: Model-specific parameters
        max_tokens: Max tokens for response
        code: The generated transform code
        explanation: The solver's explanation of the pattern
        train_examples: All training examples
        test_inputs: ALL test input grids
        test_outputs: ALL corresponding outputs from the model
        verbose: Whether to print progress

    Returns:
        Dict with 'decision' (CORRECT/WRONG/UNSURE), 'issues' (if any)
    """
    # Format training pairs
    training_pairs = ""
    training_shapes = []
    training_colors = set()

    for i, ex in enumerate(train_examples):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        training_shapes.append(f"{out.shape}")
        training_colors.update(np.unique(out).tolist())

        training_pairs += f"\nTraining Pair {i+1}:\n"
        training_pairs += f"Input ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_text(inp)}\n"
        training_pairs += f"Output ({out.shape[0]}x{out.shape[1]}):\n{grid_to_text(out)}\n"

    # Format ALL test results
    n_tests = len(test_inputs)
    test_results_str = ""
    all_output_colors = set()
    
    for i, (test_input, test_output) in enumerate(zip(test_inputs, test_outputs)):
        all_output_colors.update(np.unique(test_output).tolist())
        
        if n_tests > 1:
            test_results_str += f"\n--- Test {i+1}/{n_tests} ---\n"
        
        test_results_str += f"Input ({test_input.shape[0]}x{test_input.shape[1]}):\n"
        test_results_str += f"{grid_to_text(test_input)}\n"
        test_results_str += f"\nYour Output ({test_output.shape[0]}x{test_output.shape[1]}):\n"
        test_results_str += f"{grid_to_text(test_output)}\n"

    # Build prompt
    prompt = SELF_VERIFICATION_PROMPT.format(
        num_tests=n_tests,
        explanation=explanation[:500] if explanation else "Pattern identified from examples",
        code=code[:3000] if code else "# Code not provided",
        training_pairs=training_pairs[:4000],
        test_results=test_results_str[:4000],
        training_output_shapes=", ".join(training_shapes),
        training_output_colors=str(training_colors),
    )

    # Use HIGH reasoning for self-verification (always)
    verify_extra_body = {"reasoning": {"effort": "high"}}

    response, elapsed = await call_llm(
        model=model,
        system_prompt="You are verifying your own solution to an ARC puzzle. Be critical.",
        user_prompt=prompt,
        extra_body=verify_extra_body,
        max_tokens=max_tokens,
        temperature=0.3,
    )

    # Parse verdict and score
    result = {
        "decision": "UNSURE",
        "score": 50,  # Default score
        "feedback": "",
        "raw_response": response[:500],
    }

    response_upper = response.upper()

    # Extract VERDICT
    if "VERDICT:" in response_upper:
        verdict_idx = response_upper.find("VERDICT:")
        verdict_section = response_upper[verdict_idx:verdict_idx + 50]
        if "CORRECT" in verdict_section:
            result["decision"] = "CORRECT"
        elif "WRONG" in verdict_section:
            result["decision"] = "WRONG"
        else:
            result["decision"] = "UNSURE"
    else:
        # Fallback parsing
        if "CORRECT" in response_upper and "WRONG" not in response_upper[:response_upper.find("CORRECT") if "CORRECT" in response_upper else 0]:
            result["decision"] = "CORRECT"
        elif "WRONG" in response_upper:
            result["decision"] = "WRONG"

    # Extract SCORE
    score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
    if score_match:
        result["score"] = min(100, max(0, int(score_match.group(1))))
    else:
        # Infer score from verdict if not explicitly given
        if result["decision"] == "CORRECT":
            result["score"] = 90
        elif result["decision"] == "WRONG":
            result["score"] = 20
        else:
            result["score"] = 50

    # Extract FEEDBACK
    feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        result["feedback"] = feedback_match.group(1).strip()[:500]
    else:
        # Try to extract any explanation after verdict
        if result["decision"] == "WRONG":
            wrong_idx = response_upper.find("WRONG")
            result["feedback"] = response[wrong_idx:wrong_idx + 300].strip()
        elif result["decision"] == "UNSURE":
            unsure_idx = response_upper.find("UNSURE") if "UNSURE" in response_upper else 0
            result["feedback"] = response[unsure_idx:unsure_idx + 300].strip()

    if verbose:
        emoji = "✓" if result["decision"] == "CORRECT" else "⚠️"
        print(f"    🔍 Self-verify: {emoji} {result['decision']} (score={result['score']})")

    return result

