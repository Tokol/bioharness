from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness_config import SKILLS_DIR


@dataclass(frozen=True)
class HarnessExtension:
    name: str
    path: Path
    summary: str


def list_harness_extensions(skills_dir: Path = SKILLS_DIR) -> list[HarnessExtension]:
    """Load local skill metadata without executing any extension code."""
    if not skills_dir.exists():
        return []
    extensions: list[HarnessExtension] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        lines = [line.strip("# ").strip() for line in text.splitlines() if line.strip()]
        name = lines[0] if lines else skill_file.parent.name.replace("_", " ").title()
        summary = next((line for line in lines[1:] if not line.startswith("-")), "")
        extensions.append(HarnessExtension(name=name, path=skill_file, summary=summary))
    return extensions
