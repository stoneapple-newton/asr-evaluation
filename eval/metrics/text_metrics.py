"""WER, CER, and derived hallucination/missing-content metrics."""
from __future__ import annotations

import re
import unicodedata


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    return text


def levenshtein_counts(ref_tokens: list[str], hyp_tokens: list[str]) -> dict:
    """Return substitution / deletion / insertion counts via DP + backtrack."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = ref_tokens[i - 1] == hyp_tokens[j - 1]
            sub_cost = 0 if match else 1
            candidates = [
                (dp[i - 1][j] + 1, ("del", i - 1, j)),
                (dp[i][j - 1] + 1, ("ins", i, j - 1)),
                (dp[i - 1][j - 1] + sub_cost, ("sub" if sub_cost else "ok", i - 1, j - 1)),
            ]
            dp[i][j], bt[i][j] = min(candidates, key=lambda x: x[0])

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        op, pi, pj = bt[i][j]  # type: ignore[misc]
        if op == "sub":
            S += 1
        elif op == "del":
            D += 1
        elif op == "ins":
            I += 1
        i, j = pi, pj

    return {"substitutions": S, "deletions": D, "insertions": I, "ref_len": n}


def wer(ref_text: str, hyp_text: str) -> tuple[float, dict]:
    """Word Error Rate = (S+D+I) / N. Returns (rate, counts)."""
    ref = ref_text.split()
    hyp = hyp_text.split()
    c = levenshtein_counts(ref, hyp)
    denom = max(1, c["ref_len"])
    rate = (c["substitutions"] + c["deletions"] + c["insertions"]) / denom
    return rate, c


def cer(ref_text: str, hyp_text: str) -> tuple[float, dict]:
    """Character Error Rate — same formula applied to chars. Returns (rate, counts)."""
    ref = list(ref_text)
    hyp = list(hyp_text)
    c = levenshtein_counts(ref, hyp)
    denom = max(1, c["ref_len"])
    rate = (c["substitutions"] + c["deletions"] + c["insertions"]) / denom
    return rate, c


def derived_content_metrics(wer_counts: dict, hyp_text: str) -> dict:
    """
    Compute missing-content and hallucination proxy rates from WER alignment counts.

    missing_content_rate = D / max(1, N)
    hallucination_rate   = I / max(1, |hyp_words|)
    """
    D = wer_counts["deletions"]
    I = wer_counts["insertions"]
    N = max(1, wer_counts["ref_len"])
    hyp_len = max(1, len(hyp_text.split()))
    return {
        "missing_content_rate": D / N,
        "hallucination_rate": I / hyp_len,
    }


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_wer_stats(per_sample_wer: list[float]) -> dict:
    """Return mean, p50, p90 across a list of per-sample WER values."""
    if not per_sample_wer:
        return {"mean": None, "p50": None, "p90": None, "n": 0}
    return {
        "mean": sum(per_sample_wer) / len(per_sample_wer),
        "p50": percentile(per_sample_wer, 50),
        "p90": percentile(per_sample_wer, 90),
        "n": len(per_sample_wer),
    }
