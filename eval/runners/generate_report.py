"""
Report generator: reads run.json + summary.json + scores.jsonl and writes
report.md and report.html.  Optionally compares against a baseline summary.

Usage:
  python eval/runners/generate_report.py \\
    --run_json  results/<run_id>/run.json \\
    --summary   results/<run_id>/summary.json \\
    --scores    results/<run_id>/scores.jsonl \\
    --out_dir   results/<run_id>/

  # with baseline comparison:
  python eval/runners/generate_report.py \\
    ... \\
    --baseline_summary results/<baseline_run_id>/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from jinja2 import Template

    _JINJA2 = True
except ImportError:
    _JINJA2 = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_MD_TEMPLATE = """\
# ASR Eval Report: {{ run.run_id }}

## Run metadata
| Field | Value |
|---|---|
| Stage | {{ run.stage }} |
| Owner | {{ run.owner }} |
| Git commit | {{ run.git.commit }} ({{ run.git.branch }}) |
| Dataset | {{ run.dataset.version }} — {{ run.dataset.manifest }} |
| Started | {{ run.started_at }} |
| Ended | {{ run.ended_at }} |
| Config | {{ run.runner.config_path }} |
{% if run.notes %}| Notes | {{ run.notes }} |{% endif %}

## Summary metrics
| Metric | Value |{% if baseline_means %} Baseline | Delta |{% endif %}

|---|---|{% if baseline_means %}---|---|{% endif %}

{% for k, v in summary.means.items() %}
| {{ k }} | {{ "%.4f"|format(v) if v is not none else "n/a" }} |{% if baseline_means %} {{ "%.4f"|format(baseline_means.get(k)) if baseline_means.get(k) is not none else "n/a" }} | {{ "%+.4f"|format(v - baseline_means[k]) if (v is not none and baseline_means.get(k) is not none) else "n/a" }} |{% endif %}

{% endfor %}

## WER distribution
- Mean: {{ wer_stats.mean | round(4) if wer_stats.mean is not none else "n/a" }}
- P50:  {{ wer_stats.p50  | round(4) if wer_stats.p50  is not none else "n/a" }}
- P90:  {{ wer_stats.p90  | round(4) if wer_stats.p90  is not none else "n/a" }}
- N:    {{ wer_stats.n }}

## Worst samples by WER (top {{ worst|length }})
{% for s in worst %}
- **{{ s.sample_id }}** — WER={{ "%.3f"|format(s.metrics.wer) if s.metrics and s.metrics.wer is not none else "n/a" }} | SSS={{ "%.4f"|format(s.metrics.sss) if s.metrics and s.metrics.sss is not none else "n/a" }} | tags={{ s.failure_tags }}
{% endfor %}

## Failure tag counts
{% for tag, cnt in tag_counts.items() %}
- `{{ tag }}`: {{ cnt }}
{% endfor %}

## Errors / runner failures
Total errors: {{ summary.n_errors }}
"""

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ASR Eval — {{ run.run_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: left; }
    th { background: #f4f4f4; }
    .good  { color: #2a9d8f; }
    .warn  { color: #e9c46a; }
    .bad   { color: #e76f51; }
    .delta-pos { color: #2a9d8f; }
    .delta-neg { color: #e76f51; }
    pre { background: #f8f8f8; padding: 0.5rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>ASR Eval Report: {{ run.run_id }}</h1>

  <h2>Run metadata</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Stage</td><td>{{ run.stage }}</td></tr>
    <tr><td>Owner</td><td>{{ run.owner }}</td></tr>
    <tr><td>Git</td><td>{{ run.git.commit }} ({{ run.git.branch }})</td></tr>
    <tr><td>Dataset</td><td>{{ run.dataset.version }}</td></tr>
    <tr><td>Started</td><td>{{ run.started_at }}</td></tr>
    <tr><td>Ended</td><td>{{ run.ended_at }}</td></tr>
    {% if run.notes %}<tr><td>Notes</td><td>{{ run.notes }}</td></tr>{% endif %}
  </table>

  <h2>Summary metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th>{% if baseline_means %}<th>Baseline</th><th>Delta</th>{% endif %}</tr>
    {% for k, v in summary.means.items() %}
    <tr>
      <td>{{ k }}</td>
      <td>{{ "%.4f"|format(v) if v is not none else "n/a" }}</td>
      {% if baseline_means %}
      <td>{{ "%.4f"|format(baseline_means.get(k)) if baseline_means.get(k) is not none else "n/a" }}</td>
      <td>
        {% if v is not none and baseline_means.get(k) is not none %}
          {% set delta = v - baseline_means[k] %}
          <span class="{{ 'delta-neg' if delta > 0 else 'delta-pos' }}">
            {{ "%+.4f"|format(delta) }}
          </span>
        {% else %}n/a{% endif %}
      </td>
      {% endif %}
    </tr>
    {% endfor %}
  </table>

  <h2>WER distribution</h2>
  <ul>
    <li>Mean: {{ wer_stats.mean | round(4) if wer_stats.mean is not none else "n/a" }}</li>
    <li>P50:  {{ wer_stats.p50  | round(4) if wer_stats.p50  is not none else "n/a" }}</li>
    <li>P90:  {{ wer_stats.p90  | round(4) if wer_stats.p90  is not none else "n/a" }}</li>
    <li>N:    {{ wer_stats.n }}</li>
  </ul>

  <h2>Worst samples by WER</h2>
  <table>
    <tr><th>Sample ID</th><th>WER</th><th>SSS</th><th>Failure Tags</th></tr>
    {% for s in worst %}
    <tr>
      <td>{{ s.sample_id }}</td>
      <td class="bad">{{ "%.3f"|format(s.metrics.wer) if s.metrics and s.metrics.wer is not none else "n/a" }}</td>
      <td>{{ "%.4f"|format(s.metrics.sss) if s.metrics and s.metrics.sss is not none else "n/a" }}</td>
      <td><code>{{ s.failure_tags | join(", ") }}</code></td>
    </tr>
    {% endfor %}
  </table>

  <h2>Failure tag counts</h2>
  <table>
    <tr><th>Tag</th><th>Count</th></tr>
    {% for tag, cnt in tag_counts.items() %}
    <tr><td><code>{{ tag }}</code></td><td>{{ cnt }}</td></tr>
    {% endfor %}
  </table>

  <p>Total errors: {{ summary.n_errors }} / {{ summary.n_total }}</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dict wrapper for attribute-style access in templates
# ---------------------------------------------------------------------------

class _Obj(dict):
    def __getattr__(self, key):
        val = self.get(key)
        if isinstance(val, dict):
            return _Obj(val)
        return val


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ASR eval report generator")
    ap.add_argument("--run_json", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument(
        "--baseline_summary", default=None,
        help="Path to baseline summary.json for delta comparison",
    )
    args = ap.parse_args()

    run = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    scores = list(read_jsonl(Path(args.scores)))

    baseline_means: dict | None = None
    if args.baseline_summary:
        bsum = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        baseline_means = bsum.get("means")

    run_o = _Obj(run)
    summary_o = _Obj(summary)
    wer_stats = summary.get("wer_stats") or {}

    # Build scored-only list with attribute wrappers
    scored_only = []
    for s in scores:
        if s.get("metrics"):
            obj = _Obj(s)
            obj["metrics"] = _Obj(s["metrics"])
            scored_only.append(obj)

    worst = sorted(
        scored_only,
        key=lambda s: s.metrics.wer if s.metrics.wer is not None else 999.0,
        reverse=True,
    )[:args.top_k]

    tag_counts: dict[str, int] = {}
    for s in scores:
        for t in s.get("failure_tags") or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    ctx = dict(
        run=run_o,
        summary=summary_o,
        wer_stats=wer_stats,
        worst=worst,
        tag_counts=tag_counts,
        baseline_means=baseline_means,
    )

    out_dir = Path(args.out_dir)

    if _JINJA2:
        md = Template(_MD_TEMPLATE).render(**ctx)
        html = Template(_HTML_TEMPLATE).render(**ctx)
    else:
        # Minimal fallback without Jinja2
        lines = [
            f"# ASR Eval Report: {run.get('run_id')}",
            "",
            f"Stage: {run.get('stage')}  Owner: {run.get('owner')}",
            f"Git: {(run.get('git') or {}).get('commit')}",
            "",
            "## Summary metrics",
        ]
        for k, v in (summary.get("means") or {}).items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines += ["", "## Worst samples by WER"]
        for s in worst:
            wval = (s.get("metrics") or {}).get("wer")
            lines.append(f"- {s['sample_id']} WER={wval:.3f}" if wval is not None else f"- {s['sample_id']}")
        md = "\n".join(lines)
        html = f"<pre>{md}</pre>"

    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    # Update run.json artifact pointers
    run["artifacts"]["report_md"] = str(out_dir / "report.md")
    run["artifacts"]["report_html"] = str(out_dir / "report.html")
    Path(args.run_json).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print(f"  report.md  : {out_dir / 'report.md'}")
    print(f"  report.html: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
