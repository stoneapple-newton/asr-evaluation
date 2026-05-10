# Codex Version SSS

This directory contains the active Codex implementation of the Sentence Similarity Score project.

For the complete project guide, see the root [README.md](../README.md).

## Local Layout

```text
codex_version/
├── cli.py
├── chunker.py
├── aligner.py
├── embedder.py
├── scorer.py
├── storage.py
├── models.py
└── data/
    ├── ground_truth/
    ├── transcriptions/
    ├── runs/
    └── results/
```

## Common Commands

```powershell
uv run python -m codex_version.cli init
uv run python -m codex_version.cli new-run --run-id sample_meeting_v1 --gt sample_meeting --tx sample_meeting_v1 --location local
uv run python -m codex_version.cli run sample_meeting_v1
uv run python -m codex_version.cli history --gt sample_meeting
uv run python -m codex_version.cli show sample_meeting_v1
uv run python -m codex_version.cli compare sample_meeting_v1 sample_meeting_v2
```

Ollama must be running with an embedding model such as `nomic-embed-text`.
