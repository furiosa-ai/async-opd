"""Evaluation answer extraction and matching utilities."""

from functools import lru_cache
import os
import re


_DEFAULT_MATH_VERIFY_MAX_CHARS = 8192
_DEFAULT_FULL_RESPONSE_MATCH_MAX_CHARS = 8192


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _clean_extracted_answer(ans: str) -> str:
    """Clean answer text without destroying LaTeX tuple/list separators."""
    ans = re.sub(r'[$%]', '', ans).strip()
    # Remove thousands separators only for plain numeric answers.  Commas can
    # be semantically meaningful for MATH-500-style tuples/coordinates.
    if re.fullmatch(r'-?[\d,]+(?:\.\d+)?', ans):
        ans = ans.replace(',', '')
    return ans


def extract_answer(text: str, pattern: str = None) -> str:
    r"""Extract numeric answer from model response.

    Args:
        text: model response text
        pattern: optional regex with a capture group for the answer.
            E.g. ``"#### (\\-?[0-9\\.\\,]+)"`` for strict GSM8K format.
            If supplied, ONLY this pattern is tried (no fallback).
            If not supplied, tries the default cascade:
              1. \boxed{...}  2. ####  3. last number

    Returns:
        extracted answer string, or "" if not found
    """
    if pattern is not None:
        matches = re.findall(pattern, text)
        if matches:
            ans = _clean_extracted_answer(matches[-1].strip())
            return ans
        return ""

    # Default cascade: try \boxed{...} format (possibly nested braces)
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_matches:
        ans = boxed_matches[-1].strip()
        # Keep LaTeX commands intact for math-equivalence verification.
        ans = _clean_extracted_answer(ans)
        if ans:
            return ans
    # Try GSM8K "#### <answer>" format
    if "####" in text:
        ans = text.rsplit("####", 1)[-1].strip()
        ans = _clean_extracted_answer(ans)
        if ans:
            return ans
    # Fallback: last number in the response
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""


def score_problems(problem_results, n_samples=1):
    """Compute accuracy metrics from per-problem results.

    Args:
        problem_results: list of {"n_correct": int, "n_total": int, ...}
        n_samples: 1 for greedy accuracy, >1 for Avg@N

    Returns:
        dict with accuracy metrics (matches log.jsonl eval format).
        Empty input returns zeroed metrics (not {}) so consumers can
        distinguish "ran with no data" from "didn't run".
    """
    n_problems = len(problem_results)
    correct = sum(p["n_correct"] for p in problem_results)
    total_samples = sum(p["n_total"] for p in problem_results)
    if n_samples == 1:
        accuracy = 100.0 * correct / n_problems if n_problems > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": n_problems}
    else:
        # Include n_total==0 problems as 0 contribution (don't inflate by skipping)
        per_problem_acc = [p["n_correct"] / p["n_total"] if p["n_total"] > 0 else 0.0
                          for p in problem_results]
        avg_at_n = 100.0 * sum(per_problem_acc) / n_problems if n_problems > 0 else 0.0
        return {
            f"avg_at_{n_samples}": avg_at_n,
            "correct": correct,
            "total_samples": total_samples,
            "n_problems": n_problems,
        }


def _math_variants(text: str) -> tuple[str, ...]:
    """Generate parse variants for bare LaTeX answers.

    math-verify parses bare numeric strings well, but some bare LaTeX tuples or
    expressions need math delimiters/boxing to parse as one answer.
    """
    text = text.strip()
    if not text:
        return ()
    max_chars = _env_int("OPD_EVAL_MATH_VERIFY_MAX_CHARS",
                         _DEFAULT_MATH_VERIFY_MAX_CHARS)
    if max_chars > 0 and len(text) > max_chars:
        return ()
    variants = [text]
    if "\\" in text and not (text.startswith("$") or text.startswith("\\boxed")):
        variants.extend([f"${text}$", f"\\boxed{{{text}}}"])
    return tuple(dict.fromkeys(variants))


@lru_cache(maxsize=8192)
def _parse_math_answer(text: str):
    from math_verify import parse
    # `math_verify.parse(..., raise_on_error=False)` logs the full input on
    # timeout.  Eval responses can be tens of thousands of characters, so a
    # single symbolic fallback can otherwise dump megabytes into run.log and
    # make CPU-side scoring look hung.  Raising lets our caller handle the
    # timeout silently.
    return tuple(parse(text, raise_on_error=True))


def _answers_match_math_verify(predicted: str, gt: str) -> bool:
    try:
        from math_verify import verify
    except Exception:
        return False

    for gt_variant in _math_variants(gt):
        try:
            gt_parsed = list(_parse_math_answer(gt_variant))
        except Exception:
            continue
        if not gt_parsed:
            continue
        for pred_variant in _math_variants(predicted):
            try:
                pred_parsed = list(_parse_math_answer(pred_variant))
            except Exception:
                continue
            if pred_parsed and verify(gt_parsed, pred_parsed):
                return True
    return False


def should_try_full_response_match(ground_truth: str) -> bool:
    """Return whether full-response math parsing is worth trying.

    The full-response path is useful for symbolic answers where lightweight
    extraction may grab an inner number from a fraction/tuple/expression.  Avoid
    it for simple scalar numeric answers to keep old numeric eval behavior and
    tests stable.
    """
    if not ground_truth:
        return False
    text = ground_truth.strip()
    cleaned = _clean_extracted_answer(text)
    if re.fullmatch(r'-?\d+(?:\.\d+)?', cleaned):
        return False
    symbolic_markers = ("\\", "$", "{", "}", "(", ")", "[", "]", "/", ",", "=", "^")
    if any(marker in text for marker in symbolic_markers):
        return True
    symbolic_words = r'\b(?:sqrt|pi|infty|inf|frac|left|right|pm)\b'
    return bool(re.search(symbolic_words, text, flags=re.IGNORECASE))


def should_try_full_response_candidate(response: str, ground_truth: str) -> bool:
    """Return whether it is safe/useful to parse an entire response.

    Full-response math parsing is a fallback for short symbolic answers where
    regex extraction missed nested structure.  It is deliberately bounded: long
    model outputs can trigger repeated math-verify timeouts and huge warning
    logs while contributing little beyond the already-extracted answer.
    """
    if not response or not should_try_full_response_match(ground_truth):
        return False
    max_chars = _env_int("OPD_EVAL_FULL_RESPONSE_MATCH_MAX_CHARS",
                         _DEFAULT_FULL_RESPONSE_MATCH_MAX_CHARS)
    return max_chars > 0 and len(response) <= max_chars


def answers_match(predicted: str, gt: str, use_math_verify: bool = True) -> bool:
    """Compare predicted and ground-truth answers, normalizing numbers."""
    if not predicted or not gt:
        return False
    # Strip harmless formatting from both sides.  Preserve non-numeric commas
    # because they are meaningful in tuples/coordinates.
    gt_clean = _clean_extracted_answer(gt)
    pred_clean = _clean_extracted_answer(predicted)
    # Exact string match first
    if pred_clean == gt_clean:
        return True
    # Numeric comparison
    try:
        if abs(float(pred_clean) - float(gt_clean)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # Only invoke symbolic equivalence for symbolic ground truths.  This keeps
    # scalar integer datasets such as AIME on the legacy exact/numeric matcher.
    if use_math_verify and should_try_full_response_match(gt):
        return _answers_match_math_verify(predicted, gt)
    return False
