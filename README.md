# Sentence Similarity Score (SSS)

SSS evaluates how closely an automatic transcription matches a ground-truth transcript. The project chunks both texts into comparable segments, aligns those segments, embeds each aligned pair with Ollama, computes segment-level cosine similarity, and stores a weighted total Sentence Similarity Score.

The main implementation for this version lives in `codex_version/`.

## What This Project Does

- Stores ground-truth transcripts separately from transcription versions.
- Supports many transcription versions for one ground truth.
- Requires one YAML metadata file per transcription run/version.
- Uses Ollama embeddings, currently defaulting to `nomic-embed-text`.
- Calculates segment-level SSS and total weighted SSS.
- Saves every run as a timestamped JSON result for historical comparison.
- Provides a simple CLI for setup, running, listing, showing, and comparing results.

## Repository Layout

```text
.
├── codex_version/
│   ├── cli.py                    # Typer CLI entry point
│   ├── chunker.py                # Text splitting strategies
│   ├── aligner.py                # Segment alignment logic
│   ├── embedder.py               # Ollama embedding client
│   ├── scorer.py                 # Cosine similarity and total SSS
│   ├── storage.py                # Filesystem persistence
│   ├── models.py                 # Pydantic data models
│   ├── README.md                 # Codex-version specific notes
│   └── data/
│       ├── ground_truth/         # Ground-truth .txt files
│       ├── transcriptions/       # Transcription version .txt files
│       ├── runs/                 # Per-run YAML metadata
│       └── results/              # Timestamped JSON result history
├── kimi/                         # Earlier implementation
├── data/                         # Earlier sample data
├── pyproject.toml
└── README.md
```

## Data Model

The intended relationship is:

```text
one ground truth
  ├── transcription version 1 -> run metadata YAML -> result JSON
  ├── transcription version 2 -> run metadata YAML -> result JSON
  └── transcription version N -> run metadata YAML -> result JSON
```

Use stable IDs as file stems:

```text
codex_version/data/ground_truth/sample_meeting.txt
codex_version/data/transcriptions/sample_meeting_v1.txt
codex_version/data/transcriptions/sample_meeting_v2.txt
codex_version/data/runs/sample_meeting_v1.yaml
codex_version/data/runs/sample_meeting_v2.yaml
```

Both `sample_meeting_v1.yaml` and `sample_meeting_v2.yaml` can point to the same `ground_truth_id`, while each points to a different `transcription_id`.

## Requirements

- Python `>=3.10`
- `uv`
- Ollama running locally
- An embedding model available in Ollama

The default embedding model is:

```text
nomic-embed-text
```

Check available Ollama models:

```powershell
ollama list
```

Pull the default model if needed:

```powershell
ollama pull nomic-embed-text
```

## Installation

From the repository root:

```powershell
uv sync
```

You can run the CLI either as a module:

```powershell
uv run python -m codex_version.cli --help
```

Or through the project script:

```powershell
uv run codex-sss --help
```

## Quick Start

Create the folder structure:

```powershell
uv run python -m codex_version.cli init
```

Add a ground truth:

```text
codex_version/data/ground_truth/sample_meeting.txt
```

Add one or more transcription versions:

```text
codex_version/data/transcriptions/sample_meeting_v1.txt
codex_version/data/transcriptions/sample_meeting_v2.txt
```

Create run metadata for each transcription version:

```powershell
uv run python -m codex_version.cli new-run --run-id sample_meeting_v1 --gt sample_meeting --tx sample_meeting_v1 --location local
uv run python -m codex_version.cli new-run --run-id sample_meeting_v2 --gt sample_meeting --tx sample_meeting_v2 --location local
```

Run SSS:

```powershell
uv run python -m codex_version.cli run sample_meeting_v1
uv run python -m codex_version.cli run sample_meeting_v2
```

View history:

```powershell
uv run python -m codex_version.cli history --gt sample_meeting
```

Compare two transcription versions:

```powershell
uv run python -m codex_version.cli compare sample_meeting_v1 sample_meeting_v2
```

## CLI Commands

### `init`

Creates the expected data folders:

```powershell
uv run python -m codex_version.cli init
```

### `new-run`

Creates a YAML metadata file for a transcription version:

```powershell
uv run python -m codex_version.cli new-run --run-id sample_meeting_v1 --gt sample_meeting --tx sample_meeting_v1 --location local
```

Options:

- `--run-id`: Unique run/version ID.
- `--gt`: Ground-truth file stem from `data/ground_truth`.
- `--tx`: Transcription file stem from `data/transcriptions`.
- `--model`: Ollama embedding model. Defaults to `nomic-embed-text`.
- `--location`: Free-text source/location label.

### `run`

Computes SSS for one run metadata file:

```powershell
uv run python -m codex_version.cli run sample_meeting_v1
```

Optional model override:

```powershell
uv run python -m codex_version.cli run sample_meeting_v1 --model nomic-embed-text
```

Optional Ollama URL override:

```powershell
uv run python -m codex_version.cli run sample_meeting_v1 --ollama-url http://localhost:11434
```

### `list`

Lists stored artifacts:

```powershell
uv run python -m codex_version.cli list ground_truth
uv run python -m codex_version.cli list transcriptions
uv run python -m codex_version.cli list runs
uv run python -m codex_version.cli list results
```

### `history`

Shows saved result history:

```powershell
uv run python -m codex_version.cli history
uv run python -m codex_version.cli history --gt sample_meeting
```

### `show`

Shows segment-level scores for the latest result of a run:

```powershell
uv run python -m codex_version.cli show sample_meeting_v1
```

### `compare`

Compares two run results:

```powershell
uv run python -m codex_version.cli compare sample_meeting_v1 sample_meeting_v2
```

The command reports segment-level deltas and total SSS delta.

## Run Metadata YAML

Every transcription version should have a YAML file in:

```text
codex_version/data/runs/
```

Example:

```yaml
run_id: sample_meeting_v1
ground_truth_id: sample_meeting
transcription_id: sample_meeting_v1
model: nomic-embed-text
chunk_strategy: sentence_window
target_sentences_per_chunk: 3
max_chars_per_chunk: 900
location: local-sample
recorded_at: '2026-05-10T14:46:29'
settings:
  source: ''
  asr_model: ''
  language: ''
  prompt: ''
notes: ''
```

Important fields:

- `run_id`: Unique ID for this transcription evaluation.
- `ground_truth_id`: File stem in `data/ground_truth`.
- `transcription_id`: File stem in `data/transcriptions`.
- `model`: Ollama embedding model.
- `chunk_strategy`: Chunking method. Supported values are `sentence_window`, `sentence`, and `paragraph`.
- `target_sentences_per_chunk`: Used by `sentence_window`.
- `max_chars_per_chunk`: Maximum approximate chunk size for `sentence_window`.
- `settings`: Free-form metadata for ASR model, decoding settings, prompts, language, or other details.
- `location`: Free-form location/source label.
- `notes`: Free-form run notes.

## How SSS Is Calculated

1. The ground truth and transcription are loaded from `.txt` files.
2. Both texts are split into chunks.
3. Chunks are aligned one-to-one.
4. If one side has more chunks, adjacent short chunks are merged until counts match.
5. Each aligned ground-truth/transcription pair is embedded with Ollama.
6. Segment SSS is cosine similarity between the two embedding vectors.
7. Total SSS is a weighted average of segment scores, weighted by aligned text length.
8. A timestamped JSON result is saved under `codex_version/data/results`.

Scores are in the range `0.0` to `1.0`, where higher means the transcription segment is semantically closer to the ground-truth segment.

## Result Files

Result files are saved as JSON:

```text
codex_version/data/results/<ground_truth_id>__<transcription_id>__<run_id>__<timestamp>.json
```

Each result includes:

- Run ID
- Ground-truth ID
- Transcription ID
- Embedding model
- Chunking settings
- Total SSS
- Segment count
- Original metadata
- Segment-level aligned text and SSS

Because each run is timestamped, repeated runs are preserved for history instead of overwritten.

## Current Sample Result

Using the included sample data with `nomic-embed-text`:

```text
sample_meeting_v1 total SSS: 0.988498
sample_meeting_v2 total SSS: 0.992089
delta: +0.003591
```

Segment comparison:

```text
#   v1        v2        delta
1   1.000000  0.996414  -0.003586
2   0.996040  0.998528  +0.002488
3   0.967858  0.997627  +0.029769
4   1.000000  0.928195  -0.071805
```

## Practical Workflow

For each new transcription version:

1. Put the transcript text in `codex_version/data/transcriptions`.
2. Create a YAML file with `new-run`.
3. Edit the YAML to record model settings, prompts, language, source, and notes.
4. Run SSS.
5. Use `history` to track totals over time.
6. Use `compare` to inspect segment-level changes between versions.

## Notes and Limitations

- The current alignment strategy is simple: it aligns by chunk order and merges adjacent chunks to equalize counts.
- This works best when the transcription generally follows the same order as the ground truth.
- It is semantic similarity, not word error rate.
- High SSS does not guarantee exact wording, punctuation, speaker labels, or timestamps.
- Low segment SSS can identify areas worth manual review.

Future improvements could include dynamic-programming alignment, speaker-aware chunking, timestamp-aware alignment, embedding cache persistence, CSV export, and HTML reports.

---

## `claude_version` — Current Implementation

`claude_version` is the latest iteration of this project, available as the `claude-sss` CLI command.

### What's Different

| Feature | `codex_version` | `claude_version` |
|---|---|---|
| CLI command | `codex-sss` | `claude-sss` |
| Data location | `codex_version/data/` | `claude_version/data/` |
| `new-run` YAML | Minimal fields | Fully annotated with inline comments |
| Embedding progress | Silent | Live progress bar |
| `history` filtering | `--gt` only | `--gt` and `--tx` |
| Score colouring | None | Green ≥ 0.8 / yellow ≥ 0.5 / red < 0.5 |

### Layout

```text
claude_version/
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

### Quick Start

```powershell
# Create data folders
uv run claude-sss init

# Scaffold a run metadata YAML, then edit it
uv run claude-sss new-run --run-id meeting_v1 --gt meeting --tx meeting_v1 --location "conference_room_A"

# Run SSS
uv run claude-sss run meeting_v1

# Inspect
uv run claude-sss show meeting_v1
uv run claude-sss history --gt meeting
uv run claude-sss compare meeting_v1 meeting_v2
```

### CLI Commands

| Command    | Description                                              |
|------------|----------------------------------------------------------|
| `init`     | Create data folders                                      |
| `new-run`  | Scaffold an annotated YAML metadata file                 |
| `run`      | Run SSS pipeline and save a timestamped result           |
| `list`     | List ground_truth / transcriptions / runs / results      |
| `show`     | Segment-level SSS table for a run                        |
| `history`  | All historic results, filterable by `--gt` and `--tx`    |
| `compare`  | Segment-by-segment delta between two runs                |

### `new-run` Options

| Option       | Description                                | Default            |
|--------------|--------------------------------------------|--------------------|
| `--run-id`   | Unique run ID (required)                   |                    |
| `--gt`       | Ground-truth file stem (required)          |                    |
| `--tx`       | Transcription file stem (required)         |                    |
| `--model`    | Ollama embedding model                     | `nomic-embed-text` |
| `--strategy` | `sentence_window` \| `sentence` \| `paragraph` | `sentence_window` |
| `--location` | Where the audio was recorded               |                    |
| `--notes`    | Free-form notes                            |                    |

### Generated YAML

`new-run` writes a fully annotated YAML ready to edit:

```yaml
# SSS run metadata — edit before running
run_id: meeting_v1
ground_truth_id: meeting
transcription_id: meeting_v1

# Embedding
model: nomic-embed-text
chunk_strategy: sentence_window   # sentence_window | sentence | paragraph
target_sentences_per_chunk: 3
max_chars_per_chunk: 900

# Context
location: conference_room_A
recorded_at: null                 # ISO8601 timestamp of the recording
notes: null

# ASR / transcription settings — fill in as needed
settings:
  asr_model: ''           # e.g. whisper-large-v3
  language: ''            # e.g. en
  prompt: ''              # prompt given to ASR, if any
  audio_source: ''        # e.g. microphone, phone_call, video_conference
```

### Result Files

Result files are saved as JSON under `claude_version/data/results/`:

```text
{ground_truth_id}__{transcription_id}__{run_id}__{timestamp}.json
```

Each run appends a new file, so the full history is always preserved.
