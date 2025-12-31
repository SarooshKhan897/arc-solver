"""Run module - orchestrates solving tasks."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MAX_WORKERS, MIN_SOLUTIONS_REQUIRED, MODEL_RANK, MIN_CONFIDENCE_SCORE
from src.models import ArcEvaluator, TaskResult, SolutionCandidate
from src.perception import perceive_batch, difference_batch, perceive_task, analyze_task
from src.solve import solve_with_models
from src.utils.trace import TRACE_LOGGER


def select_by_score_and_rank(candidates: list[SolutionCandidate]) -> list[SolutionCandidate]:
    """
    Sort candidates by score (highest first), then by model rank as tiebreaker.
    """
    def sort_key(c: SolutionCandidate) -> tuple[int, int]:
        # Negative score for descending order, positive rank for ascending
        try:
            model_rank = MODEL_RANK.index(c.model_id)
        except ValueError:
            model_rank = len(MODEL_RANK)  # Unknown models go last
        return (-c.verifier_score, model_rank)
    
    return sorted(candidates, key=sort_key)


# =============================================================================
# Single Task Solver
# =============================================================================

async def solve_task(
    task_data: dict[str, Any],
    verbose: bool = True,
) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """
    Solve a single ARC task with potentially multiple test inputs.

    Flow:
    1. Perceive all grids (training + ALL test inputs)
    2. Compute deltas for training pairs
    3. Run multiple models in parallel
    4. Return predictions for ALL test inputs (2 attempts each)

    Args:
        task_data: Task with 'train' and 'test' keys
        verbose: Whether to print progress

    Returns:
        (predictions_per_test, info_dict)
        predictions_per_test[i] = [attempt1, attempt2] for test_input[i]
    """
    TRACE_LOGGER.reset()

    train = task_data['train']
    test_cases = task_data['test']
    test_inputs = [np.array(t['input']) for t in test_cases]
    n_examples = len(train)
    n_tests = len(test_inputs)

    if verbose:
        print("=" * 60)
        print("🎭 ARC SOLVER")
        print("=" * 60)
        print(f"   Training examples: {n_examples}")
        print(f"   Test inputs: {n_tests}")
        print("-" * 60)

    # Step 1: Perceive all grids (training + ALL test inputs)
    if verbose:
        print("  👁️ Perceiving grids...")

    all_grids = []
    for pair in train:
        all_grids.append(np.array(pair['input']))
        all_grids.append(np.array(pair['output']))
    # Add ALL test inputs
    for ti in test_inputs:
        all_grids.append(ti)

    all_perceptions = await perceive_batch(all_grids, verbose=verbose)

    # Organize perceptions
    perceptions = []
    for i in range(n_examples):
        perceptions.append({
            'input': all_perceptions[i * 2],
            'output': all_perceptions[i * 2 + 1],
        })
    
    # Test perceptions - ALL of them
    test_perceptions = all_perceptions[n_examples * 2:]

    if verbose:
        print(f"     ✓ Perceived {len(all_grids)} grids ({n_examples * 2} training + {n_tests} test)")

    # Step 2: Compute deltas
    if verbose:
        print("  🔍 Computing deltas...")

    pairs = [(np.array(p['input']), np.array(p['output'])) for p in train]
    perc_pairs = [(perceptions[i]['input'], perceptions[i]['output']) for i in range(n_examples)]
    deltas = await difference_batch(pairs, perc_pairs, verbose=verbose)

    if verbose:
        print(f"     ✓ Computed {len(deltas)} deltas")

    # Step 3: Perceiver generates transformation hypotheses
    if verbose:
        print("  🔮 Perceiver analyzing task for hypotheses...")

    task_perception = await perceive_task(task_data, verbose=verbose)
    hypotheses = task_perception.get("transformation_hypotheses", [])

    if verbose:
        print(f"     ✓ Perceiver generated {len(hypotheses)} transformation hypotheses")
        if task_perception.get("key_insight"):
            print(f"     💡 Key insight: {task_perception['key_insight'][:60]}...")

    # Step 4: Solve with models (exhaustive primary + smart fallback)
    if verbose:
        print("-" * 60)
        print("  🚀 Running exhaustive model execution...")

    candidates = await solve_with_models(
        task_data=task_data,
        perceptions=perceptions,
        deltas=deltas,
        test_perception=test_perceptions,  # Pass ALL test perceptions
        hypotheses=hypotheses,
        observations=task_perception.get("observations"),
        key_insight=task_perception.get("key_insight"),
        target_solutions=MIN_SOLUTIONS_REQUIRED,  # Need 2 solutions for ARC
        verbose=verbose,
    )

    # Step 5: Organize predictions per test
    # candidates is already sorted by score (top solutions)
    
    if not candidates:
        # No solutions at all - return test inputs as fallback
        predictions_per_test = [[ti, ti] for ti in test_inputs]
        info = {
            'source': 'no_solution',
            'candidates_found': 0,
            'n_tests': n_tests,
            'trace_summary': TRACE_LOGGER.get_summary(),
        }
    else:
        # Get top 2 candidates (already sorted by score)
        top_candidates = candidates[:2]
        if len(top_candidates) == 1:
            top_candidates = [top_candidates[0], top_candidates[0]]
        
        # Determine source based on scores
        high_conf = [c for c in candidates if c.verifier_score >= MIN_CONFIDENCE_SCORE]
        source = 'high_confidence' if high_conf else 'best_available'
        
        # Organize predictions: for each test input, get [attempt1, attempt2]
        predictions_per_test = []
        for test_idx in range(n_tests):
            attempts = []
            for cand in top_candidates:
                if cand.test_results and test_idx < len(cand.test_results):
                    attempts.append(cand.test_results[test_idx])
                else:
                    attempts.append(test_inputs[test_idx])
            predictions_per_test.append(attempts)

        best = candidates[0]
        info = {
            'source': source,
            'candidates_found': len(candidates),
            'high_confidence_count': len(high_conf),
            'n_tests': n_tests,
            'winner_model': best.model_id,
            'winner_score': best.verifier_score,
            'winner_self_verify': best.self_verify_decision,
            'models_with_solutions': list(set(c.model_id for c in candidates)),
            'trace_summary': TRACE_LOGGER.get_summary(),
        }

    if verbose:
        print("-" * 60)
        print(f"  📊 Final: {len(candidates)} candidate(s), {n_tests} test output(s)")
        if candidates:
            high_conf_count = len([c for c in candidates if c.verifier_score >= MIN_CONFIDENCE_SCORE])
            source_emoji = "🔥" if high_conf_count >= 2 else "✅" if high_conf_count >= 1 else "⚠️"
            print(f"  {source_emoji} High-confidence solutions: {high_conf_count}/{len(candidates)}")
            for i, c in enumerate(candidates[:2]):
                conf = "🔥" if c.verifier_score >= MIN_CONFIDENCE_SCORE else "✅"
                print(f"     {i+1}. {conf} [{c.model_id}]: score={c.verifier_score}")

    return predictions_per_test, info


# =============================================================================
# Task Evaluation
# =============================================================================

async def evaluate_task(
    task_path: str,
    task_idx: int = 0,
    total_tasks: int = 1,
    verbose: bool = True,
) -> TaskResult:
    """
    Load, solve, and evaluate a single task with multiple test inputs.
    Uses official ARC scoring: score = correct_tests / total_tests

    Args:
        task_path: Path to task JSON file
        task_idx: Index for progress display
        total_tasks: Total tasks for progress display
        verbose: Whether to print progress

    Returns:
        TaskResult with predictions and fractional evaluation
    """
    task_name = os.path.basename(task_path)

    if verbose:
        print(f"\n🚀 [{task_idx + 1}/{total_tasks}] Starting: {task_name}")

    try:
        # Load task
        with open(task_path) as f:
            task_data = json.load(f)

        # Get ALL test cases
        test_cases = task_data['test']
        n_tests = len(test_cases)
        
        # Check for ground truths
        has_ground_truth = 'output' in test_cases[0]
        ground_truths = []
        if has_ground_truth:
            ground_truths = [np.array(t['output']) for t in test_cases]

        # Prepare for solver - pass ALL test inputs (without outputs)
        task_for_solver = {
            'train': task_data['train'],
            'test': [{'input': t['input']} for t in test_cases],
        }

        if verbose and n_tests > 1:
            print(f"   📋 Task has {n_tests} test inputs")

        # Solve - now returns predictions for ALL test inputs
        predictions_per_test, info = await solve_task(task_for_solver, verbose=verbose)

        # Evaluate using official ARC scoring
        n_correct = 0
        per_test_correct = []
        score = 0.0

        if ground_truths:
            n_correct, n_total, score, per_test_correct = ArcEvaluator.evaluate_task(
                predictions_per_test, ground_truths
            )

            if verbose:
                if n_tests == 1:
                    status = "✅ CORRECT" if n_correct == 1 else "❌ WRONG"
                    print(f"\n  📊 Result: {status}")
                else:
                    print(f"\n  📊 Result: {n_correct}/{n_tests} test cases correct ({score*100:.1f}%)")
                    for i, correct in enumerate(per_test_correct):
                        status = "✅" if correct else "❌"
                        print(f"     Test {i+1}: {status}")

        return TaskResult(
            task_id=task_name,
            predictions=[[p.tolist() for p in attempts] for attempts in predictions_per_test],
            n_test_cases=n_tests,
            n_correct=n_correct,
            score=score,
            per_test_correct=per_test_correct,
            solve_info=info,
        )

    except Exception as e:
        if verbose:
            print(f"  ❌ Error: {str(e)[:100]}")
        return TaskResult(
            task_id=task_name,
            predictions=[],
            error=str(e)[:200],
        )


# =============================================================================
# Batch Runner
# =============================================================================

async def run_tasks(
    task_paths: list[str],
    max_concurrent: int | None = None,
    verbose: bool = True,
) -> list[TaskResult]:
    """
    Run multiple tasks with concurrency control.
    Uses official ARC scoring across all test cases.

    Args:
        task_paths: List of task file paths
        max_concurrent: Max concurrent tasks (defaults to MAX_WORKERS)
        verbose: Whether to print progress

    Returns:
        List of TaskResult objects
    """
    if max_concurrent is None:
        max_concurrent = MAX_WORKERS

    total = len(task_paths)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(path: str, idx: int) -> TaskResult:
        async with semaphore:
            return await evaluate_task(path, idx, total, verbose=verbose)

    if verbose:
        print(f"🚀 Running {total} tasks (max {max_concurrent} concurrent)")

    # Disable trace logging for batch runs
    TRACE_LOGGER.disable()

    tasks = [
        asyncio.create_task(bounded_task(path, i))
        for i, path in enumerate(task_paths)
    ]

    results: list[TaskResult] = []
    completed = 0
    total_score = 0.0  # Accumulated score (n_correct/n_test_cases per task)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        
        # Score per task: proportional (e.g., 1/3 correct = 0.33, 2/2 = 1.0)
        total_score += result.score  # score = n_correct / n_test_cases

        if verbose and (completed % 5 == 0 or completed == total):
            pct = total_score / completed * 100 if completed > 0 else 0
            print(f"📊 Progress: {completed}/{total} tasks, "
                  f"{total_score:.1f}/{completed} score ({pct:.1f}%)")

    TRACE_LOGGER.enable()

    return results


def run_tasks_sync(task_paths: list[str], **kwargs) -> list[TaskResult]:
    """Synchronous wrapper for run_tasks."""
    return asyncio.get_event_loop().run_until_complete(
        run_tasks(task_paths, **kwargs)
    )


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_evaluation(
    data_dir: str = "ARC-AGI-2/data/evaluation",
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run evaluation on ARC tasks using official ARC scoring.

    Args:
        data_dir: Directory containing task JSON files
        limit: Max number of tasks to run (None = all)
        verbose: Whether to print progress

    Returns:
        Summary dict with accuracy and results
    """
    import glob

    task_paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if limit:
        task_paths = task_paths[:limit]

    results = await run_tasks(task_paths, verbose=verbose)

    # Calculate summary - official ARC scoring
    total_tasks = len(results)
    total_test_cases = sum(r.n_test_cases for r in results)
    total_correct_tests = sum(r.n_correct for r in results)
    tasks_fully_correct = sum(1 for r in results if r.score == 1.0)
    errors = sum(1 for r in results if r.error)
    
    # Per-task average score (official ARC metric)
    avg_score = sum(r.score for r in results) / total_tasks if total_tasks > 0 else 0

    summary = {
        "total_tasks": total_tasks,
        "total_test_cases": total_test_cases,
        "total_correct_tests": total_correct_tests,
        "tasks_fully_correct": tasks_fully_correct,
        "errors": errors,
        "test_accuracy": total_correct_tests / total_test_cases if total_test_cases > 0 else 0,
        "task_avg_score": avg_score,
        "results": [r.model_dump() for r in results],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS (Official ARC Scoring)")
        print("=" * 60)
        print(f"  📋 Tasks evaluated: {total_tasks}")
        print(f"  📋 Total test cases: {total_test_cases}")
        print(f"  ✅ Test cases correct: {total_correct_tests}/{total_test_cases} ({summary['test_accuracy']*100:.1f}%)")
        print(f"  🎯 Tasks fully solved: {tasks_fully_correct}/{total_tasks}")
        print(f"  📊 Average task score: {avg_score*100:.1f}%")
        print(f"  ❌ Errors: {errors}")

    return summary

