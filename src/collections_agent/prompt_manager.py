"""Prompt versioning, loading, and rollback.

Live prompts are Markdown files in ``prompts/``.  Each saved version is
written to ``data/prompt_versions/{agent}/{timestamp}.md`` with a JSON
sidecar holding evaluation data, enabling a full audit trail.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = _PROJECT_ROOT / "prompts"
VERSIONS_DIR = _PROJECT_ROOT / "data" / "prompt_versions"


@dataclass
class PromptVersion:
    version_id: str
    agent_name: str
    prompt_text: str
    created_at: str
    eval_data: dict = field(default_factory=dict)
    is_active: bool = False


# ---------------------------------------------------------------------------

class PromptManager:
    """Load, version, and rollback agent prompts."""

    PROMPT_FILES: dict[str, str] = {
        "agent1": "agent1_assessment.md",
        "agent2": "agent2_resolution_voice.md",
        "agent3": "agent3_final_notice.md",
    }

    def __init__(self) -> None:
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for agent in self.PROMPT_FILES:
            (VERSIONS_DIR / agent).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_prompt(self, agent_name: str) -> str:
        """Return the currently active prompt text for *agent_name*."""
        active = self._get_active_version(agent_name)
        if active is not None:
            return active.prompt_text

        # Fall back to the original prompt file on disk.
        filename = self.PROMPT_FILES.get(agent_name)
        if filename is None:
            raise ValueError(f"Unknown agent: {agent_name}")
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text()

    def list_versions(self, agent_name: str) -> list[PromptVersion]:
        """List all saved versions for *agent_name* (newest first)."""
        agent_dir = VERSIONS_DIR / agent_name
        if not agent_dir.exists():
            return []

        versions: list[PromptVersion] = []
        for meta_path in sorted(agent_dir.glob("*.json"), reverse=True):
            meta = json.loads(meta_path.read_text())
            prompt_path = agent_dir / f"{meta['version_id']}.md"
            versions.append(
                PromptVersion(
                    version_id=meta["version_id"],
                    agent_name=meta["agent_name"],
                    prompt_text=prompt_path.read_text() if prompt_path.exists() else "",
                    created_at=meta["created_at"],
                    eval_data=meta.get("eval_data", {}),
                    is_active=meta.get("is_active", False),
                )
            )
        return versions

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_version(
        self,
        agent_name: str,
        prompt_text: str,
        eval_data: dict,
        *,
        activate: bool = False,
    ) -> str:
        """Persist a new prompt version.  Returns the version ID."""
        now = datetime.now(timezone.utc)
        version_id = now.strftime("%Y%m%dT%H%M%SZ")
        agent_dir = VERSIONS_DIR / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt text
        (agent_dir / f"{version_id}.md").write_text(prompt_text)

        # Write metadata sidecar
        meta = {
            "version_id": version_id,
            "agent_name": agent_name,
            "created_at": now.isoformat(),
            "eval_data": eval_data,
            "is_active": activate,
        }
        (agent_dir / f"{version_id}.json").write_text(json.dumps(meta, indent=2))

        if activate:
            self._set_active(agent_name, version_id)

        logger.info("Saved prompt version %s for %s (active=%s)", version_id, agent_name, activate)
        return version_id

    def rollback(self, agent_name: str, version_id: str) -> None:
        """Activate a previous prompt version (rollback)."""
        prompt_path = VERSIONS_DIR / agent_name / f"{version_id}.md"
        if not prompt_path.exists():
            raise ValueError(f"Version {version_id} not found for {agent_name}")
        self._set_active(agent_name, version_id)
        logger.info("Rolled back %s to version %s", agent_name, version_id)

    # ------------------------------------------------------------------
    # Protected-section enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def extract_protected_sections(prompt_text: str) -> list[str]:
        """Return all ``<!-- PROTECTED -->...<!-- /PROTECTED -->`` blocks."""
        import re

        pattern = r"<!--\s*PROTECTED\s*-->(.+?)<!--\s*/PROTECTED\s*-->"
        return re.findall(pattern, prompt_text, re.DOTALL)

    @classmethod
    def validate_protected_preserved(cls, original: str, candidate: str) -> bool:
        """Ensure every protected section in *original* exists in *candidate*."""
        for section in cls.extract_protected_sections(original):
            if section.strip() not in candidate:
                return False
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_active_version(self, agent_name: str) -> PromptVersion | None:
        active_file = VERSIONS_DIR / agent_name / "active.txt"
        if not active_file.exists():
            return None
        version_id = active_file.read_text().strip()
        prompt_path = VERSIONS_DIR / agent_name / f"{version_id}.md"
        if not prompt_path.exists():
            return None
        return PromptVersion(
            version_id=version_id,
            agent_name=agent_name,
            prompt_text=prompt_path.read_text(),
            created_at="",
            is_active=True,
        )

    def _set_active(self, agent_name: str, version_id: str) -> None:
        agent_dir = VERSIONS_DIR / agent_name
        # Update all sidecar files
        for meta_path in agent_dir.glob("*.json"):
            meta = json.loads(meta_path.read_text())
            meta["is_active"] = meta["version_id"] == version_id
            meta_path.write_text(json.dumps(meta, indent=2))
        # Write pointer file
        (agent_dir / "active.txt").write_text(version_id)
