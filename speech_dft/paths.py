"""Project locations; commands can be run from the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "results" / "experiments"
COMPARISONS_DIR = ROOT / "results" / "comparisons"
TEACHING_DIR = ROOT / "results" / "teaching"
