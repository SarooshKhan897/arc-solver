"""Differencer - compares input/output pairs to extract transformation deltas."""

import json
import re
from typing import Any

import numpy as np

from src.config import Role, get_role_model, get_role_extra_body
from src.llms.client import call_llm
from src.perception.objects import compare_grids_fast
from src.utils.grid import grid_to_text

# =============================================================================
# System Prompt
# =============================================================================

DIFFERENCER_SYSTEM = """You are the DIFFERENCER specialist in an ARC puzzle solving system.
Your ONLY job is to describe WHAT CHANGED between input and output - NO hypothesis about WHY.

OUTPUT FORMAT (JSON):
{
  "object_changes": [
    {"action": "moved", "object": "blue rectangle", "from": "top-left", "to": "bottom-right", "delta": [3, 2]},
    {"action": "recolored", "object": "all red pixels", "from_color": "red", "to_color": "blue"},
    {"action": "deleted", "object": "small green square"},
    {"action": "created", "object": "new yellow pixel", "location": "center"},
    {"action": "scaled", "object": "blue shape", "factor": 2},
    {"action": "rotated", "object": "L-shape", "degrees": 90},
    ...
  ],
  "structural_changes": [
    "grid expanded from 3x3 to 6x6",
    "objects merged into single shape",
    "pattern was tiled 2x2",
    ...
  ],
  "preserved": [
    "background color (black)",
    "relative positions of objects",
    "color palette",
    ...
  ],
  "summary": "Brief one-sentence summary of the transformation"
}

Be PRECISE and SPECIFIC. Focus on OBSERVABLE changes, not interpretations."""


# =============================================================================
# Differencer Function
# =============================================================================

async def difference(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_perception: dict[str, Any] | None = None,
    output_perception: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Call Differencer to compare input/output and extract delta.

    Args:
        input_grid: The input grid
        output_grid: The output grid
        input_perception: Optional pre-computed perception of input
        output_perception: Optional pre-computed perception of output
        verbose: Whether to print progress

    Returns:
        Structured delta dictionary
    """
    # Get code-based delta
    code_delta = compare_grids_fast(input_grid, output_grid)

    # Build prompt
    user_prompt = f"""Compare these INPUT and OUTPUT grids. What changed?

INPUT GRID ({input_grid.shape[0]}x{input_grid.shape[1]}):
{grid_to_text(input_grid)}

OUTPUT GRID ({output_grid.shape[0]}x{output_grid.shape[1]}):
{grid_to_text(output_grid)}

CODE-DETECTED CHANGES:
- Size change: {code_delta.size_change}
- Color changes: {code_delta.color_changes}
- Preserved: {code_delta.constants}
"""

    # Add perception context if available
    if input_perception:
        user_prompt += f"""
INPUT PERCEPTION:
{json.dumps(input_perception, indent=2)[:2000]}
"""

    if output_perception:
        user_prompt += f"""
OUTPUT PERCEPTION:
{json.dumps(output_perception, indent=2)[:2000]}
"""

    user_prompt += "\nProvide your structured JSON analysis of what changed between input and output."

    # Call LLM
    model = get_role_model(Role.DIFFERENCER)
    extra_body = get_role_extra_body(Role.DIFFERENCER)
    response, elapsed = await call_llm(
        model=model,
        system_prompt=DIFFERENCER_SYSTEM,
        user_prompt=user_prompt,
        extra_body=extra_body,
        temperature=0.3,
    )

    # Try to parse JSON response
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            delta = json.loads(json_match.group())
            if verbose:
                n_changes = len(delta.get('object_changes', []))
                print(f"     Differencer: {n_changes} object changes detected")
            return delta
    except json.JSONDecodeError:
        if verbose:
            print("     Differencer: JSON parse failed, using code fallback")

    # Fallback to code-based delta
    fallback = {
        "object_changes": [],
        "structural_changes": [],
        "preserved": code_delta.constants,
        "summary": f"Size change: {code_delta.size_change}",
    }

    # Include raw model response for downstream use
    if response and response.strip():
        fallback["raw_model_analysis"] = response[:3000]

    return fallback


async def difference_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    perceptions: list[tuple[dict, dict]] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compute deltas for multiple input/output pairs in parallel."""
    import asyncio

    if perceptions is None:
        perceptions = [(None, None)] * len(pairs)

    tasks = [
        difference(inp, out, inp_perc, out_perc, verbose=False)
        for (inp, out), (inp_perc, out_perc) in zip(pairs, perceptions)
    ]
    results = await asyncio.gather(*tasks)

    if verbose:
        print(f"     ✓ Computed {len(results)} deltas")

    return results

