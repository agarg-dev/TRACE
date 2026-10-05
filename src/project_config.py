"""Repository paths, model paths, and shared layer defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL = "Qwen3-8B"

DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

ACTIVATION_CACHE_DIR = OUTPUT_DIR / "activations"
RUNS_DIR = OUTPUT_DIR / "runs"
VQ_RUNS_DIR = RUNS_DIR / "vq"
DETECTION_RUNS_DIR = RUNS_DIR / "detection"
STEERING_RUNS_DIR = RUNS_DIR / "steering"

HARMBENCH_MODEL_DIR = MODEL_DIR / "HarmBench-Llama-2-13b-cls"

# Default Qwen layer pair. Other generators pass their layers explicitly.
READ_LAYER = 20
TARGET_LAYER = 24
