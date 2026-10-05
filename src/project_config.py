"""Project paths and shared layer settings."""

from pathlib import Path

from data.dataset_splits import DEFAULT_DATASET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL = "Qwen3-8B"

DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
ACTIVATION_CACHE_DIR = OUTPUT_DIR / "activations"
RUNS_DIR = OUTPUT_DIR / "runs"
VQ_RUNS_DIR = RUNS_DIR / "vq"
DETECTION_RUNS_DIR = RUNS_DIR / "detection"
DATA_DIR = DATA_ROOT / DEFAULT_DATASET
HARMBENCH_MODEL_DIR = MODEL_DIR / "HarmBench-Llama-2-13b-cls"

READ_LAYER = 20
TARGET_LAYER = 24


def resolve_project_path(path):
    """Resolve a project-relative path without depending on the caller's working directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_path(dataset=DEFAULT_DATASET):
    return DATA_ROOT / dataset


def base_model_path(base_model=DEFAULT_BASE_MODEL):
    return MODEL_DIR / base_model
