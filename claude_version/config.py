from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TRANSCRIPTION_DIR = DATA_DIR / "transcriptions"
RUNS_DIR = DATA_DIR / "runs"
RESULTS_DIR = DATA_DIR / "results"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_STRATEGY = "sentence_window"
DEFAULT_TARGET_SENTENCES = 3
DEFAULT_MAX_CHARS = 900
