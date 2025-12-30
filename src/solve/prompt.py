"""Prompt generation for the solver."""

import json
from typing import Any

import numpy as np

from src.utils.grid import grid_to_text
from src.perception.analyzer import analyze_example, analyze_grid, analyze_task
from src.perception.perceiver import format_hypotheses_for_solver


# =============================================================================
# System Prompt
# =============================================================================

SOLVER_SYSTEM = """You are an expert ARC-AGI puzzle solver. Your task is to:
1. Analyze the training examples to find the transformation pattern
2. Write Python code that implements this transformation

CRITICAL REQUIREMENTS:
- Your code MUST work for ALL training examples
- The transformation must be general, not hardcoded to specific examples
- Use numpy for grid operations

OUTPUT FORMAT:
1. Brief explanation of the pattern (2-3 sentences)
2. Python code in a ```python block with a `transform(grid)` function

The transform function signature:
```python
import numpy as np

def transform(grid):
    # grid is a 2D numpy array
    # Return the transformed grid as a 2D numpy array
    ...
    return result
```"""


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_prompt(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
    include_analysis: bool = True,
) -> str:
    """
    Generate a rich prompt for the solver.

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Pre-computed perceptions for each training pair
        deltas: Pre-computed deltas for each training pair
        test_perception: Perception of the test input
        hypotheses: Pre-computed transformation hypotheses (top 5)
        feedback: Optional feedback from previous failed attempt
        include_analysis: Whether to include detailed grid analysis

    Returns:
        The complete prompt string
    """
    parts = []
    train_examples = task_data['train']

    # Header
    parts.append("=" * 60)
    parts.append("ARC PUZZLE - TRAINING EXAMPLES")
    parts.append("=" * 60)

    # Training examples
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        parts.append(f"\n{'='*60}")
        parts.append(f"TRAINING PAIR {idx + 1}")
        parts.append(f"{'='*60}")

        # Grid representation
        parts.append(f"\n--- INPUT ({inp.shape[0]}x{inp.shape[1]}) ---")
        parts.append(grid_to_text(inp))

        # Enhanced grid analysis
        if include_analysis:
            example_analysis = analyze_example(inp, out, idx + 1)
            inp_a = example_analysis.input_analysis
            out_a = example_analysis.output_analysis
            trans_a = example_analysis.transform_analysis
            
            parts.append(f"\n📊 INPUT STATS:")
            parts.append(f"  Grid Size: {inp_a.rows} × {inp_a.cols} ({inp_a.total_cells} cells)")
            parts.append(f"  Colors Used: {inp_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(inp_a.color_palette)}")
            parts.append(f"  Background: {inp_a.background_color}")
            parts.append(f"  Fill Ratio: {inp_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {inp_a.shapes_by_color}")

        parts.append(f"\n--- OUTPUT ({out.shape[0]}x{out.shape[1]}) ---")
        parts.append(grid_to_text(out))

        if include_analysis:
            parts.append(f"\n📊 OUTPUT STATS:")
            parts.append(f"  Grid Size: {out_a.rows} × {out_a.cols} ({out_a.total_cells} cells)")
            parts.append(f"  Colors Used: {out_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(out_a.color_palette)}")
            parts.append(f"  Background: {out_a.background_color}")
            parts.append(f"  Fill Ratio: {out_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {out_a.shapes_by_color}")
            
            parts.append(f"\n🔄 KEY TRANSFORMATIONS:")
            parts.append(f"  Size Change: {trans_a.size_change_cells} cells ({trans_a.size_change_percent:+.1f}%)")
            parts.append(f"  New Colors Introduced: {'YES' if trans_a.new_colors_introduced else 'NO'}")
            parts.append(f"  Density Change: {trans_a.density_change_percent:+.1f}%")
            parts.append(f"  Shape Count: {trans_a.input_shape_count} → {trans_a.output_shape_count}")
            parts.append(f"  Transform Hints: {', '.join(trans_a.hints)}")

        # Add perception if available
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            if 'input' in perc and 'output' in perc:
                parts.append("\n--- PERCEPTION ANALYSIS ---")
                parts.append(f"Input objects: {len(perc['input'].get('objects', []))}")
                parts.append(f"Output objects: {len(perc['output'].get('objects', []))}")

        # Add delta if available
        if deltas and idx < len(deltas):
            delta = deltas[idx]
            parts.append("\n--- TRANSFORMATION DELTA ---")
            if delta.get('summary'):
                parts.append(f"Summary: {delta['summary']}")
            if delta.get('object_changes'):
                parts.append(f"Changes: {json.dumps(delta['object_changes'][:5], indent=2)}")

    # Cross-example pattern summary
    if include_analysis:
        task_analysis = analyze_task(task_data, "current")
        parts.append("\n" + "=" * 60)
        parts.append("📋 CROSS-EXAMPLE PATTERNS")
        parts.append("=" * 60)
        parts.append(f"  Size always preserved: {task_analysis.consistent_size_preservation}")
        parts.append(f"  Colors always preserved: {task_analysis.consistent_color_preservation}")
        parts.append(f"  Shape count preserved: {task_analysis.consistent_shape_count}")
        if task_analysis.common_hints:
            parts.append(f"  Common hints: {', '.join(task_analysis.common_hints)}")

    # Transformation hypotheses from Perceiver (if provided)
    if hypotheses:
        parts.append("\n" + format_hypotheses_for_solver(hypotheses))

    # Size pattern summary
    parts.append("\n" + "=" * 60)
    parts.append("SIZE PATTERN SUMMARY")
    parts.append("=" * 60)

    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        parts.append(f"  Example {idx+1}: {inp.shape} → {out.shape}")

    # Test inputs - ALL of them!
    test_inputs = task_data['test']
    n_tests = len(test_inputs)
    
    parts.append("\n" + "=" * 60)
    parts.append(f"TEST INPUTS ({n_tests} total)")
    parts.append("=" * 60)
    
    for test_idx, test_case in enumerate(test_inputs):
        test_input = np.array(test_case['input'])
        
        if n_tests > 1:
            parts.append(f"\n--- TEST INPUT {test_idx + 1}/{n_tests} ({test_input.shape[0]}x{test_input.shape[1]}) ---")
        else:
            parts.append(f"\n--- TEST INPUT ({test_input.shape[0]}x{test_input.shape[1]}) ---")
        parts.append(grid_to_text(test_input))

        # Test input analysis
        if include_analysis:
            test_a = analyze_grid(test_input)
            parts.append(f"\n📊 TEST INPUT {test_idx + 1} STATS:")
            parts.append(f"  Grid Size: {test_a.rows} × {test_a.cols} ({test_a.total_cells} cells)")
            parts.append(f"  Colors Used: {test_a.colors_used} colors")
            parts.append(f"  Color Palette: {', '.join(test_a.color_palette)}")
            parts.append(f"  Background: {test_a.background_color}")
            parts.append(f"  Fill Ratio: {test_a.fill_ratio:.1f}%")
            parts.append(f"  Shapes by Color: {test_a.shapes_by_color}")

    # Test perceptions (if provided as list)
    if test_perception:
        # Handle both single perception and list of perceptions
        if isinstance(test_perception, list):
            for i, tp in enumerate(test_perception):
                parts.append(f"\n--- TEST {i+1} PERCEPTION ---")
                parts.append(f"Objects: {len(tp.get('objects', []))}")
                if tp.get('global_patterns'):
                    parts.append(f"Patterns: {tp['global_patterns']}")
        else:
            parts.append("\n--- TEST INPUT PERCEPTION ---")
            parts.append(f"Objects: {len(test_perception.get('objects', []))}")
            if test_perception.get('global_patterns'):
                parts.append(f"Patterns: {test_perception['global_patterns']}")

    # Feedback from previous attempt
    if feedback:
        parts.append("\n" + "=" * 60)
        parts.append("⚠️ FEEDBACK FROM PREVIOUS ATTEMPT")
        parts.append("=" * 60)
        parts.append(feedback)

    # Task instructions
    parts.append("""

============================================================
YOUR TASK
============================================================

1. Study the training examples above
2. Find the SINGLE rule that transforms ALL inputs to outputs
3. Write a `transform(grid)` function that implements this rule

Provide:
1. A brief explanation of the pattern (2-3 sentences)
2. Python code in a ```python block with the `transform` function
""")

    return '\n'.join(parts)


def generate_feedback_prompt(
    original_prompt: str,
    code: str,
    feedback_messages: list[str],
    attempt_num: int,
) -> str:
    """
    Generate a prompt with feedback for retry attempts.

    Args:
        original_prompt: The original task prompt
        code: The code that failed
        feedback_messages: Error messages from the failed attempt
        attempt_num: Current attempt number

    Returns:
        Updated prompt with feedback
    """
    feedback_section = f"""

============================================================
⚠️ ATTEMPT {attempt_num} FAILED - FEEDBACK
============================================================

Your previous code:
```python
{code}
```

Errors:
{chr(10).join(feedback_messages)}

Please fix these issues and provide corrected code.
"""

    return original_prompt + feedback_section

