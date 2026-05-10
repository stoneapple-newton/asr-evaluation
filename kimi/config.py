"""Configuration defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TRANSCRIPTION_DIR = DATA_DIR / "transcriptions"
RUNS_DIR = DATA_DIR / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

DEFAULT_CHUNK_STRATEGY = "sentence"
