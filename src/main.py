"""Main entry point for the ARC Solver."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Jupyter notebook async compatibility
import nest_asyncio
nest_asyncio.apply()

# Try to use uvloop for better performance
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

from src.models import ArcEvaluator
from src.run import solve_task, evaluate_task, run_tasks, run_evaluation


# =============================================================================
# Single Task Runner
# =============================================================================

async def run_single_task(task_path: str, verbose: bool = True) -> dict:
    """
    Run the solver on a single task.

    Args:
        task_path: Path to task JSON file
        verbose: Whether to print detailed output

    Returns:
        Result dict with predictions and evaluation
    """
    # Load task
    with open(task_path) as f:
        task_data = json.load(f)

    task_name = os.path.basename(task_path)

    # Check for ground truth
    has_ground_truth = 'output' in task_data['test'][0]
    ground_truth = np.array(task_data['test'][0]['output']) if has_ground_truth else None

    # Prepare for solver
    task_for_solver = {
        'train': task_data['train'],
        'test': [{'input': task_data['test'][0]['input']}],
    }

    print("=" * 60)
    print(f"📋 Task: {task_name}")
    print(f"   Training examples: {len(task_data['train'])}")
    print(f"   Ground truth: {'✓ Available' if has_ground_truth else '✗ Not available'}")
    print("=" * 60)

    start_time = datetime.now()

    # Solve
    predictions, info = await solve_task(task_for_solver, verbose=verbose)

    duration = (datetime.now() - start_time).total_seconds()

    # Evaluate
    result = {
        "task": task_name,
        "predictions": [p.tolist() for p in predictions],
        "info": info,
        "duration_seconds": duration,
    }

    if ground_truth is not None:
        exact_match, accuracy = ArcEvaluator.evaluate_predictions(predictions, ground_truth)
        result["exact_match"] = exact_match
        result["accuracy"] = accuracy

        print("\n" + "=" * 60)
        print("📊 EVALUATION")
        print("=" * 60)
        print(f"   Prediction 1: {'✅ CORRECT' if np.array_equal(predictions[0], ground_truth) else '❌ Wrong'}")
        print(f"   Prediction 2: {'✅ CORRECT' if np.array_equal(predictions[1], ground_truth) else '❌ Wrong'}")
        print(f"\n   Overall: {'🎉 SUCCESS' if exact_match else '❌ FAILED'}")
        print(f"   ⏱️  Time: {duration:.1f}s")

    return result


def run_single_task_sync(task_path: str, verbose: bool = True) -> dict:
    """Synchronous wrapper for run_single_task."""
    return asyncio.get_event_loop().run_until_complete(
        run_single_task(task_path, verbose=verbose)
    )


# =============================================================================
# Random Task Runner
# =============================================================================

async def run_random_tasks(
    data_dir: str = "ARC-AGI-2/data/evaluation",
    num_tasks: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run random tasks from a data directory.

    Args:
        data_dir: Directory containing task JSON files
        num_tasks: Number of random tasks to run
        verbose: Whether to print progress

    Returns:
        Summary dict
    """
    import glob
    import random

    task_paths = glob.glob(os.path.join(data_dir, "*.json"))
    random.shuffle(task_paths)
    selected = task_paths[:num_tasks]

    print("=" * 60)
    print(f"🎲 RANDOM ARC EVALUATION ({num_tasks} tasks)")
    print("=" * 60)
    print(f"📋 Selected: {[os.path.basename(p) for p in selected[:5]]}...")

    start_time = datetime.now()

    results = await run_tasks(selected, verbose=verbose)

    duration = (datetime.now() - start_time).total_seconds()

    # Summary
    correct = sum(1 for r in results if r.exact_match)
    errors = sum(1 for r in results if r.error)

    print("\n" + "=" * 60)
    print("🏆 RESULTS")
    print("=" * 60)
    print(f"🎯 Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"❌ Errors: {errors}")
    print(f"⏱️  Total time: {duration:.0f}s ({duration/len(results):.1f}s per task avg)")

    return {
        "total": len(results),
        "correct": correct,
        "errors": errors,
        "accuracy": correct / len(results) if results else 0,
        "duration": duration,
        "results": [r.model_dump() for r in results],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ARC Solver - Multi-model ARC-AGI solver")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single task
    single_parser = subparsers.add_parser("solve", help="Solve a single task")
    single_parser.add_argument("task_path", help="Path to task JSON file")
    single_parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    # Random tasks
    random_parser = subparsers.add_parser("random", help="Run random tasks")
    random_parser.add_argument("-n", "--num", type=int, default=10, help="Number of tasks")
    random_parser.add_argument("-d", "--data-dir", default="ARC-AGI-2/data/evaluation", help="Data directory")
    random_parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    # Full evaluation
    eval_parser = subparsers.add_parser("eval", help="Run full evaluation")
    eval_parser.add_argument("-d", "--data-dir", default="ARC-AGI-2/data/evaluation", help="Data directory")
    eval_parser.add_argument("-l", "--limit", type=int, help="Max tasks to run")
    eval_parser.add_argument("-o", "--output", help="Output JSON file for results")
    eval_parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.command == "solve":
        result = run_single_task_sync(args.task_path, verbose=not args.quiet)
        print(f"\n✓ Completed: {result.get('exact_match', 'N/A')}")

    elif args.command == "random":
        asyncio.get_event_loop().run_until_complete(
            run_random_tasks(args.data_dir, args.num, verbose=not args.quiet)
        )

    elif args.command == "eval":
        summary = asyncio.get_event_loop().run_until_complete(
            run_evaluation(args.data_dir, args.limit, verbose=not args.quiet)
        )
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Results saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

