import os
from pathlib import Path


def _slug_to_title(name: str) -> str:
    s = name.replace("-", " ").replace("_", " ").replace(".", " ")
    s = " ".join(s.split())
    return s.title() if s else name


def _infer_repo_title() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return _slug_to_title(root.name)
    except Exception:
        return "Data Analysis Studio"


DEFAULT_NAME = "Data Analysis Studio"
DEFAULT_TAGLINE = "확장 가능한 EDA·모델링·프로파일링·시각화 플랫폼"

APP_NAME = os.getenv("APP_NAME") or _infer_repo_title() or DEFAULT_NAME
TAGLINE = os.getenv("APP_TAGLINE") or DEFAULT_TAGLINE

