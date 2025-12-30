"""Core solver - single model call with feedback loop."""

from typing import Any

import numpy as np

from src.config import SOLVER_MODELS, MODEL_RANK, MIN_CONFIDENCE_SCORE
from src.llms.client import call_llm
from src.models import SolutionCandidate
from src.solve.executor import execute_transform, parse_llm_response, test_on_examples
from src.solve.prompt import SOLVER_SYSTEM, generate_prompt, generate_feedback_prompt
from src.verification.self_verifier import self_verify
from src.utils.trace import TRACE_LOGGER

# Fixed self-verification model (always gpt-5.2 with high reasoning)
SELF_VERIFY_MODEL = "openai/gpt-5.2"
SELF_VERIFY_EXTRA_BODY = {"reasoning": {"effort": "high"}}


# =============================================================================
# Single Model Solver
# =============================================================================

async def solve_single(
    task_data: dict[str, Any],
    model_config: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    max_tries: int | None = None,
    verbose: bool = True,
) -> tuple[SolutionCandidate | None, SolutionCandidate | None]:
    """
    Solve a task using a single model with retry loop.

    This is the core solving function - it:
    1. Generates a prompt
    2. Calls the model
    3. Tests on training examples
    4. If passed, self-verifies the output (always using gpt-5.2)
    5. If failed, retries with feedback

    Args:
        task_data: Task with 'train' and 'test' keys
        model_config: Model configuration dict with 'id', 'model', 'extra_body', etc.
        perceptions: Pre-computed perceptions
        deltas: Pre-computed deltas
        test_perception: Perception of test input
        hypotheses: Pre-computed transformation hypotheses (top 5)
        max_tries: Override default tries from config
        verbose: Whether to print progress

    Returns:
        Tuple of (self_verified_solution, training_passed_solution)
        - self_verified_solution: Solution that passed self-verification, or None
        - training_passed_solution: Best solution that passed training (even if self-verify failed), or None
    """
    model_id = model_config["id"]
    model = model_config["model"]
    extra_body = model_config.get("extra_body")
    max_tokens = model_config.get("max_tokens")
    tries = max_tries or model_config.get("tries", 5)

    train_examples = task_data['train']
    # Get ALL test inputs
    test_inputs = [np.array(t['input']) for t in task_data['test']]
    n_tests = len(test_inputs)

    # Generate initial prompt
    prompt = generate_prompt(
        task_data=task_data,
        perceptions=perceptions,
        deltas=deltas,
        test_perception=test_perception,
        hypotheses=hypotheses,
    )

    if verbose:
        print(f"     🚀 [{model_id}] Starting ({tries} tries)...")

    attempt_history = []
    # Track best training-passed solution (even if self-verify fails)
    best_training_passed: SolutionCandidate | None = None

    for attempt in range(1, tries + 1):
        try:
            # Call the model
            response, elapsed = await call_llm(
                model=model,
                system_prompt=SOLVER_SYSTEM,
                user_prompt=prompt,
                extra_body=extra_body,
                max_tokens=max_tokens,
                temperature=0.7,
            )

            TRACE_LOGGER.log(f"solver_{model_id}", model, prompt[:500], response[:500], elapsed)

            # Parse response
            parsed = parse_llm_response(response)
            if not parsed['code']:
                if verbose:
                    print(f"     [{model_id}] Attempt {attempt}/{tries}: ✗ No code found")
                prompt = generate_feedback_prompt(prompt, "", ["No valid code found"], attempt)
                continue

            # Test on training examples
            all_passed, feedback_messages = test_on_examples(parsed['code'], train_examples)

            if not all_passed:
                if verbose:
                    print(f"     [{model_id}] Attempt {attempt}/{tries}: ✗ Failed training")
                attempt_history.append({
                    "attempt": attempt,
                    "reason": "training_failed",
                    "feedback": feedback_messages,
                })
                prompt = generate_feedback_prompt(
                    prompt, parsed['code'], feedback_messages, attempt
                )
                continue

            # Passed training! Now self-verify
            if verbose:
                print(f"     [{model_id}] Attempt {attempt}/{tries}: ✓ Passed training")

            # Execute on ALL test inputs
            test_results = []
            for ti in test_inputs:
                result = execute_transform(parsed['code'], ti)
                test_results.append(result)
            
            if verbose and n_tests > 1:
                print(f"     [{model_id}] Applied transform to {n_tests} test inputs")

            # Save as training-passed candidate (in case self-verify fails)
            # We'll get the score from self-verification below
            # For now, create a placeholder that will be updated after self-verify
            current_code = parsed['code']
            current_explanation = parsed['explanation'] or ''

            # Self-verify using FIXED model (gpt-5.2 with high reasoning)
            # This also gives us a score for ranking
            if verbose:
                print(f"     [{model_id}] 🔍 Self-verification with gpt-5.2 ({n_tests} test(s))...")

            sv_result = await self_verify(
                model=SELF_VERIFY_MODEL,  # Always use gpt-5.2
                model_id="gpt-5.2",
                extra_body=SELF_VERIFY_EXTRA_BODY,  # Always high reasoning
                max_tokens=None,  # Let model decide
                code=current_code,  # Include the generated code
                explanation=current_explanation,
                train_examples=train_examples,
                test_inputs=test_inputs,  # ALL test inputs
                test_outputs=test_results,  # ALL test outputs
                verbose=verbose,
            )

            sv_decision = sv_result.get('decision', 'UNSURE')
            sv_score = sv_result.get('score', 50)
            sv_feedback = sv_result.get('feedback', '')

            # Always save as training-passed candidate with the score (for fallback)
            if best_training_passed is None or sv_score > best_training_passed.verifier_score:
                training_passed_candidate = SolutionCandidate(
                    code=current_code,
                    explanation=current_explanation,
                    model_id=model_id,
                    verifier_score=sv_score,
                    verifier_verdict=sv_decision,
                    self_verify_decision="TRAINING_PASSED",
                    attempts=attempt,
                    test_results=test_results,
                )
                best_training_passed = training_passed_candidate

            if sv_decision != 'CORRECT':
                if verbose:
                    print(f"     [{model_id}] 🔍 Self-verify: ⚠️ {sv_decision} (score={sv_score})")
                prompt = generate_feedback_prompt(
                    prompt,
                    current_code,
                    [f"Self-verification: {sv_decision}. {sv_feedback}"],
                    attempt,
                )
                continue

            # SUCCESS! Self-verify passed with score
            if verbose:
                print(f"     [{model_id}] ✅ Self-verify=CORRECT (score={sv_score})")
                if n_tests > 1:
                    print(f"     [{model_id}] Generated {n_tests} test outputs")

            self_verified_solution = SolutionCandidate(
                code=current_code,
                explanation=current_explanation,
                model_id=model_id,
                verifier_score=sv_score,
                verifier_verdict="APPROVED",
                self_verify_decision=sv_decision,
                attempts=attempt,
                test_results=test_results,
            )
            return (self_verified_solution, best_training_passed)

        except Exception as e:
            if verbose:
                print(f"     [{model_id}] Attempt {attempt}/{tries}: ✗ Error - {str(e)[:50]}")
            attempt_history.append({
                "attempt": attempt,
                "reason": "error",
                "error": str(e)[:100],
            })

    if verbose:
        print(f"     [{model_id}] ❌ Exhausted {tries} tries")

    # Return None for self-verified, but may have training-passed fallback
    return (None, best_training_passed)


# =============================================================================
# Multi-Model Parallel Solver
# =============================================================================

async def solve_with_models(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    models: list[dict[str, Any]] | None = None,
    stop_on_success: int = 2,
    verbose: bool = True,
) -> tuple[list[SolutionCandidate], list[SolutionCandidate]]:
    """
    Solve a task using multiple models in parallel.

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Pre-computed perceptions
        deltas: Pre-computed deltas
        test_perception: Perception of test input
        hypotheses: Pre-computed transformation hypotheses (top 5)
        models: List of model configs (defaults to SOLVER_MODELS)
        stop_on_success: Stop when this many solutions found from different models
        verbose: Whether to print progress

    Returns:
        Tuple of (self_verified_candidates, training_passed_candidates)
        - self_verified_candidates: Solutions that passed self-verification
        - training_passed_candidates: Solutions that passed training (fallback)
    """
    import asyncio

    if models is None:
        models = SOLVER_MODELS

    self_verified: list[SolutionCandidate] = []
    training_passed: list[SolutionCandidate] = []
    models_with_solution: set[str] = set()
    stop_event = asyncio.Event()
    lock = asyncio.Lock()

    async def run_model(model_config: dict[str, Any]) -> None:
        """Run a single model and add results to candidates."""
        if stop_event.is_set():
            return

        sv_result, tp_result = await solve_single(
            task_data=task_data,
            model_config=model_config,
            perceptions=perceptions,
            deltas=deltas,
            test_perception=test_perception,
            hypotheses=hypotheses,
            verbose=verbose,
        )

        async with lock:
            # Always track training-passed solutions (even if self-verify succeeded)
            if tp_result and tp_result.model_id not in [c.model_id for c in training_passed]:
                training_passed.append(tp_result)

            # Track self-verified solutions
            if sv_result:
                if sv_result.model_id not in models_with_solution:
                    self_verified.append(sv_result)
                    models_with_solution.add(sv_result.model_id)

                    is_high_confidence = sv_result.verifier_score >= MIN_CONFIDENCE_SCORE
                    confidence_tag = "🔥" if is_high_confidence else "✅"
                    
                    if verbose:
                        print(f"  {confidence_tag} [{sv_result.model_id}] Self-verified solution! "
                              f"(score={sv_result.verifier_score})")

                    # Check stop condition: need N solutions with score >= MIN_CONFIDENCE_SCORE
                    high_confidence_solutions = [c for c in self_verified if c.verifier_score >= MIN_CONFIDENCE_SCORE]
                    if len(high_confidence_solutions) >= stop_on_success:
                        if verbose:
                            print(f"  🎯 Got {stop_on_success} high-confidence ({MIN_CONFIDENCE_SCORE}+) solutions!")
                        stop_event.set()

    # Launch all models in parallel
    if verbose:
        print(f"  🚀 Launching {len(models)} models in parallel...")

    tasks = [asyncio.create_task(run_model(cfg)) for cfg in models]
    await asyncio.gather(*tasks, return_exceptions=True)

    if verbose:
        if self_verified:
            high_conf = [c for c in self_verified if c.verifier_score >= MIN_CONFIDENCE_SCORE]
            print(f"  ✓ Finished with {len(self_verified)} self-verified solution(s) "
                  f"({len(high_conf)} high-confidence {MIN_CONFIDENCE_SCORE}+)")
        elif training_passed:
            best_score = max(c.verifier_score for c in training_passed)
            print(f"  ⚠️ No self-verified solutions, but {len(training_passed)} training-passed fallback(s) "
                  f"(best score={best_score})")
        else:
            print(f"  ❌ No solutions found")

    return (self_verified, training_passed)

