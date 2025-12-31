"""Perceiver - structured grid analysis using LLM."""

import json
import re
from typing import Any

import numpy as np

from src.config import Role, get_role_model, get_role_extra_body
from src.llms.client import call_llm
from src.perception.objects import perceive_grid_fast, ObjectPreprocessor
from src.utils.grid import grid_to_text
from src.models import color_name

# =============================================================================
# System Prompt
# =============================================================================

PERCEIVER_SYSTEM = """You are the PERCEIVER specialist in an ARC puzzle solving system.
Your ONLY job is to describe what you see in the grid - NO hypothesis generation.

OUTPUT FORMAT (JSON):
{
  "objects": [
    {"id": 1, "color": "blue", "shape": "rectangle", "size": 4, "position": "top-left", "special": ""},
    ...
  ],
  "relationships": [
    {"type": "adjacent", "obj1": 1, "obj2": 2, "direction": "right"},
    {"type": "contained", "outer": 1, "inner": 2},
    {"type": "aligned", "objects": [1, 2, 3], "axis": "horizontal"},
    ...
  ],
  "global_patterns": [
    "grid has horizontal symmetry",
    "objects form a 2x2 repeating pattern",
    ...
  ],
  "notable_features": [
    "single red pixel at center",
    "blue L-shaped object in corner",
    ...
  ]
}

Be PRECISE and EXHAUSTIVE. Describe EVERYTHING you observe.
Do NOT suggest what the transformation might be - just describe the grid."""


# =============================================================================
# Perceiver Function
# =============================================================================

async def perceive(
    grid: np.ndarray,
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    """
    Call Perceiver to get structured grid representation.

    Args:
        grid: The grid to analyze
        verbose: Whether to print progress
        max_retries: Number of retry attempts on JSON parse failure
        retry_delay: Delay in seconds between retries

    Returns:
        Structured perception dictionary
    """
    import asyncio
    
    # Get code-based perception as fallback
    code_perception = perceive_grid_fast(grid)

    # Get all patterns for context
    all_patterns = {
        "tiling": ObjectPreprocessor.detect_tiling(grid),
        "frame": ObjectPreprocessor.detect_frame(grid),
    }

    # Build prompt
    user_prompt = f"""Analyze this grid and provide structured JSON output.

GRID ({grid.shape[0]}x{grid.shape[1]}):
{grid_to_text(grid)}

CODE-DETECTED OBJECTS ({len(code_perception.objects)} found):
{json.dumps([obj.to_dict() for obj in code_perception.objects[:10]], indent=2)}

CODE-DETECTED PATTERNS:
- Symmetry: {code_perception.symmetry}
- Tiling: {all_patterns['tiling']}
- Frame: {all_patterns['frame']}

Provide your complete JSON analysis. Include any objects, relationships, or patterns the code may have missed."""

    # Call LLM with retry logic for JSON parse failures
    model = get_role_model(Role.PERCEIVER)
    extra_body = get_role_extra_body(Role.PERCEIVER)
    
    last_response = ""
    for attempt in range(max_retries):
        response, elapsed = await call_llm(
            model=model,
            system_prompt=PERCEIVER_SYSTEM,
            user_prompt=user_prompt,
            extra_body=extra_body,
            temperature=0.3,
        )
        last_response = response

        # Try to parse JSON response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                perception = json.loads(json_match.group())
                if verbose:
                    n_objects = len(perception.get('objects', []))
                    n_relations = len(perception.get('relationships', []))
                    print(f"     Perceiver: {n_objects} objects, {n_relations} relations")
                return perception
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"     Perceiver: JSON parse failed, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                if verbose:
                    print("     Perceiver: JSON parse failed after retries, using code fallback")

    # Fallback to code-based perception
    global_patterns_list = []
    if code_perception.symmetry.get("horizontal"):
        global_patterns_list.append("grid has horizontal symmetry")
    if code_perception.symmetry.get("vertical"):
        global_patterns_list.append("grid has vertical symmetry")
    if all_patterns['tiling'].get('is_tiled'):
        global_patterns_list.append("objects form a tiled/repeating pattern")
    if code_perception.patterns:
        global_patterns_list.extend(code_perception.patterns)
    
    fallback = {
        "objects": [
            {
                "id": i,
                "color": color_name(obj.color),
                "shape": "rectangle" if obj.is_rectangle else "irregular",
                "size": obj.size,
                "position": f"row {obj.bounding_box[0]}-{obj.bounding_box[2]}, col {obj.bounding_box[1]}-{obj.bounding_box[3]}",
            }
            for i, obj in enumerate(code_perception.objects)
        ],
        "relationships": [],
        "global_patterns": global_patterns_list,
        "notable_features": [],
        "tiling": all_patterns['tiling'],
        "frame": all_patterns['frame'],
    }

    # Include raw model response for downstream use
    if last_response and last_response.strip():
        fallback["raw_model_analysis"] = last_response

    return fallback


async def perceive_batch(
    grids: list[np.ndarray],
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Perceive multiple grids in parallel."""
    import asyncio
    tasks = [perceive(grid, verbose=False) for grid in grids]
    results = await asyncio.gather(*tasks)
    if verbose:
        print(f"     ✓ Perceived {len(results)} grids")
    return results


# =============================================================================
# Task-Level Perception with Transformation Hypotheses
# =============================================================================

TASK_PERCEIVER_SYSTEM = """You are the PERCEIVER specialist in an ARC puzzle solving system.

Your job is to analyze ALL training examples together and identify:
1. What objects/patterns exist in the grids
2. What transformations occur between input and output
3. Generate 5 POSSIBLE TRANSFORMATION HYPOTHESES (ranked by likelihood)

OUTPUT FORMAT (JSON):
{
  "observations": {
    "common_input_features": ["list of features seen in all inputs"],
    "common_output_features": ["list of features seen in all outputs"],
    "size_pattern": "same size | grows | shrinks | varies",
    "color_changes": "none | recoloring | new colors added | colors removed"
  },
  "transformation_hypotheses": [
    {
      "rank": 1,
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "rule": "Clear, specific description of the transformation rule",
      "evidence": "How this explains all training examples"
    },
    {
      "rank": 2,
      "confidence": "...",
      "rule": "...",
      "evidence": "..."
    },
    ... (exactly 5 hypotheses, ranked from most to least likely)
  ],
  "key_insight": "The single most important observation about this puzzle"
}

RULES FOR HYPOTHESES:
- Each hypothesis must be DIFFERENT - explore various interpretations
- Be SPECIFIC: "move objects" is bad, "move each colored shape 2 cells right" is good
- Rank 1 = your best guess, Rank 5 = least likely but still plausible
- Use evidence from multiple examples to support each hypothesis"""


async def perceive_task(
    task_data: dict[str, Any],
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    """
    Perceive an entire task and generate transformation hypotheses.
    
    This analyzes ALL training examples together to identify the transformation rule.
    
    Args:
        task_data: Task with 'train' and 'test' keys
        verbose: Whether to print progress
        max_retries: Number of retry attempts on JSON parse failure
        retry_delay: Delay in seconds between retries
        
    Returns:
        Dict with 'observations', 'transformation_hypotheses', 'key_insight'
    """
    import asyncio
    
    train_examples = task_data['train']
    
    # Build prompt with all examples
    prompt_parts = []
    prompt_parts.append("Analyze these training examples to identify the transformation rule.\n")
    
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        prompt_parts.append(f"\n{'='*50}")
        prompt_parts.append(f"EXAMPLE {idx + 1}")
        prompt_parts.append(f"{'='*50}")
        
        prompt_parts.append(f"\nINPUT ({inp.shape[0]}x{inp.shape[1]}):")
        prompt_parts.append(grid_to_text(inp))
        
        prompt_parts.append(f"\nOUTPUT ({out.shape[0]}x{out.shape[1]}):")
        prompt_parts.append(grid_to_text(out))
        
        # Quick stats
        inp_colors = set(np.unique(inp).tolist())
        out_colors = set(np.unique(out).tolist())
        prompt_parts.append(f"\nStats: Input colors={inp_colors}, Output colors={out_colors}")
        prompt_parts.append(f"Size change: {inp.shape} → {out.shape}")
    
    # Add test input for context
    test_input = np.array(task_data['test'][0]['input'])
    prompt_parts.append(f"\n{'='*50}")
    prompt_parts.append("TEST INPUT (for context)")
    prompt_parts.append(f"{'='*50}")
    prompt_parts.append(f"\nTEST ({test_input.shape[0]}x{test_input.shape[1]}):")
    prompt_parts.append(grid_to_text(test_input))
    
    prompt_parts.append("""
    
Now analyze all examples and output your JSON with:
1. observations (common patterns)
2. transformation_hypotheses (exactly 5, ranked)
3. key_insight (most important observation)
""")
    
    user_prompt = '\n'.join(prompt_parts)
    
    # Call LLM
    model = get_role_model(Role.PERCEIVER)
    extra_body = get_role_extra_body(Role.PERCEIVER)
    
    if verbose:
        print("  👁️ Perceiver analyzing task...")
    
    # Parse response with retry logic
    result = {
        "observations": {},
        "transformation_hypotheses": [],
        "key_insight": "",
        "raw_response": "",
    }
    
    for attempt in range(max_retries):
        response, elapsed = await call_llm(
            model=model,
            system_prompt=TASK_PERCEIVER_SYSTEM,
            user_prompt=user_prompt,
            extra_body=extra_body,
            temperature=0.4,  # Slight creativity for diverse hypotheses
        )
        
        result["raw_response"] = response
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                result["observations"] = parsed.get("observations", {})
                result["transformation_hypotheses"] = parsed.get("transformation_hypotheses", [])[:5]
                result["key_insight"] = parsed.get("key_insight", "")
                
                if verbose:
                    n_hyp = len(result["transformation_hypotheses"])
                    print(f"     ✓ Generated {n_hyp} transformation hypotheses")
                    if result["key_insight"]:
                        insight = result["key_insight"][:60]
                        print(f"     💡 Key insight: {insight}...")
                
                return result
                        
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"     ⚠️ JSON parse failed, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                if verbose:
                    print("     ⚠️ JSON parse failed after retries, extracting from text...")
                # Try to extract hypotheses from text
                result["transformation_hypotheses"] = _extract_hypotheses_text(response)
    
    return result


def _extract_hypotheses_text(text: str) -> list[dict[str, Any]]:
    """Extract hypotheses from unstructured text."""
    hypotheses = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for numbered items
        if re.match(r'^[1-5][\.\):]', line):
            hypotheses.append({
                "rank": len(hypotheses) + 1,
                "confidence": "MEDIUM",
                "rule": line.lstrip('0123456789.)]: '),
                "evidence": "",
            })
    
    return hypotheses


def format_hypotheses_for_solver(hypotheses: list[dict[str, Any]]) -> str:
    """Format hypotheses for inclusion in solver prompt."""
    if not hypotheses:
        return ""
    
    lines = [
        "=" * 60,
        "🔮 TRANSFORMATION HYPOTHESES (from Perceiver)",
        "=" * 60,
    ]
    
    for h in hypotheses:
        rank = h.get("rank", "?")
        conf = h.get("confidence", "?")
        rule = h.get("rule", "No rule")
        evidence = h.get("evidence", "")
        
        lines.append(f"\n#{rank} [{conf}]: {rule}")
        if evidence:
            lines.append(f"   Evidence: {evidence}")
    
    return '\n'.join(lines)

