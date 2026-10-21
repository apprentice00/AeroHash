"""Repository-owned defaults; user paths remain relative to the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
