"""
Compare two evaluation run summaries and print a pass/fail gate report.

Usage:
  python scripts/compare_runs.py \\
    --baseline  results/<baseline_run_id>/summary.json \\
    --candidate results/<candidate_run_id>/summary.json

Exit codes:
  0 — all gates passed
  1 — one or more gates failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default pass/fail thresholds (absolute deltas — positive means regression)
# ---------------------------------------------------------------------------
DEFAULT_GATES = {
    "wer":                  {"max_delta": 0.03,  "direction": "increase"},
    "cer":                  {"max_delta": 0.03,  "direction": "increase"},
    "missing_content_rate": {"max_delta": 0.01,  "direction": "increase"},
    "hallucination_rate":   {"max_delta": 0.01,  "direction": "increase"},
    "der":                  {"max_delta": 0.02,  "direction": "increase"},
    "rtf":                  {"max_delta": 0.5,   "direction": "increase"},
}


def load_summary(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(baseline: dict, candidate: dict, gates: dict) -> list[dict]:
    """Return a list of gate results with pass/fail status."""
    base_means = baseline.get("means") or {}
    cand_means = candidate.get("means") or {}
    results = []

    for metric, gate in gates.items():
        base_val = base_means.get(metric)
        cand_val = cand_means.get(metric)

        if base_val is None or cand_val is None:
            results.append({
                "metric": metric,
                "status": "skip",
                "baseline": base_val,
                "candidate": cand_val,
                "delta": None,
                "threshold": gate["max_delta"],
                "reason": "metric absent in one or both summaries",
            })
            continue

        delta = cand_val - base_val
        if gate["direction"] == "increase":
            passed = delta <= gate["max_delta"]
        else:
            passed = delta >= -gate["max_delta"]

        results.append({
            "metric": metric,
            "status": "pass" if passed else "FAIL",
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta,
            "threshold": gate["max_delta"],
            "reason": None,
        })

    return results


def print_report(results: list[dict], baseline_path: str, candidate_path: str) -> bool:
    """Print a formatted comparison table. Returns True if all gates passed."""
    print(f"\nBaseline : {baseline_path}")
    print(f"Candidate: {candidate_path}\n")

    col_w = [24, 10, 10, 10, 10, 8]
    headers = ["Metric", "Baseline", "Candidate", "Delta", "Threshold", "Status"]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("-" * sum(col_w))

    all_passed = True
    for r in results:
        status = r["status"]
        if status == "FAIL":
            all_passed = False
        delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "n/a"
        base_str  = f"{r['baseline']:.4f}"  if r["baseline"]  is not None else "n/a"
        cand_str  = f"{r['candidate']:.4f}" if r["candidate"] is not None else "n/a"
        thr_str   = f"{r['threshold']:.4f}" if r["threshold"] is not None else "n/a"
        row = [r["metric"], base_str, cand_str, delta_str, thr_str, status]
        print("".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print()
    if all_passed:
        print("Result: ALL GATES PASSED")
    else:
        failed = [r["metric"] for r in results if r["status"] == "FAIL"]
        print(f"Result: FAILED — {len(failed)} gate(s) failed: {', '.join(failed)}")

    return all_passed


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two ASR eval run summaries")
    ap.add_argument("--baseline",  required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument(
        "--gates", default=None,
        help="Path to a JSON file with custom gate definitions (optional)",
    )
    args = ap.parse_args()

    baseline  = load_summary(args.baseline)
    candidate = load_summary(args.candidate)

    gates = DEFAULT_GATES
    if args.gates:
        gates = json.loads(Path(args.gates).read_text())

    results = compare(baseline, candidate, gates)
    passed = print_report(results, args.baseline, args.candidate)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
