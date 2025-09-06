import os
from pathlib import Path


def _slug_to_title(name: str) -> str:
    s = name.replace("-", " ").replace("_", " ").replace(".", " ")
    s = " ".join(s.split())
    return s.title() if s else name


def _infer_repo_title() -> str:
    """Infer a human-friendly project title.

    Priority:
    1) Repository name from .git/config (remote url)
    2) Current folder name (project root)
    3) Hardcoded default
    """
    try:
        # settings.py lives under <project_root>/src/settings.py
        # so project root is two levels up from this file
        root = Path(__file__).resolve().parents[1]

        # 1) Try to read repo name from git config
        git_config = root / ".git" / "config"
        if git_config.exists():
            try:
                text = git_config.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("url ="):
                        url = line.split("=", 1)[1].strip()
                        # Extract last path segment as repo name and strip .git
                        tail = url.rsplit("/", 1)[-1]
                        if ":" in tail and "/" not in tail:
                            # Handle uncommon edge, generally rsplit('/') covers ssh urls too
                            tail = tail.split(":", 1)[-1]
                        name = tail[:-4] if tail.endswith(".git") else tail
                        title = _slug_to_title(name)
                        if title:
                            return title
            except Exception:
                pass

        # 2) Fallback to project root folder name
        return _slug_to_title(root.name)
    except Exception:
        # 3) Final fallback
        return "Data Analysis Studio"


DEFAULT_NAME = "Data Analysis Studio"
DEFAULT_TAGLINE = "확장 가능한 EDA·모델링·프로파일링·시각화 플랫폼"

APP_NAME = os.getenv("APP_NAME") or _infer_repo_title() or DEFAULT_NAME
TAGLINE = os.getenv("APP_TAGLINE") or DEFAULT_TAGLINE
