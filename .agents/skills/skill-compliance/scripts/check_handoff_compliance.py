#!/usr/bin/env python3
"""Check agent-handoff skill complies with the handoff protocol in AGENTS.md.

Verifies:
  - .handoffs/ directory exists at project root
  - agent-handoff/SKILL.md exists
  - AGENTS.md has a Handoff Protocol section
  - AGENTS.md mentions compaction relationship
  - agent-handoff SKILL.md references .handoffs/ with sub-second timestamps
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _resolve_project_root(skills_dir: str) -> Path:
    """Walk up from skills dir to find project root (contains AGENTS.md)."""
    p = Path(skills_dir).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "AGENTS.md").exists():
            return parent
    return p.parent  # fallback


def _check_file_exists(path: Path, label: str) -> tuple[bool, str]:
    exists = path.exists()
    return exists, f"{label}: {'found' if exists else 'MISSING'} at {path}"


def _check_contains(path: Path, pattern: str, label: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Cannot check {label}: file missing"
    text = path.read_text(encoding="utf-8")
    found = re.search(pattern, text, re.IGNORECASE) is not None
    return found, f"{label}: {'found' if found else 'NOT FOUND'} pattern /{pattern}/"


def run(skills_dir: str) -> dict:
    root = _resolve_project_root(skills_dir)
    agents_md = root / "AGENTS.md"
    handoffs_dir = root / ".handoffs"
    handoff_skill = Path(skills_dir) / "agent-handoff" / "SKILL.md"

    details = []
    all_pass = True

    # 1. .handoffs/ directory exists
    ok, msg = _check_file_exists(handoffs_dir, ".handoffs/ directory")
    details.append({"check_item": "handoffs-dir-exists", "passed": ok, "evidence": msg})
    all_pass = all_pass and ok

    # 2. agent-handoff SKILL.md exists
    ok, msg = _check_file_exists(handoff_skill, "agent-handoff/SKILL.md")
    details.append({"check_item": "handoff-skill-exists", "passed": ok, "evidence": msg})
    all_pass = all_pass and ok

    if ok and handoff_skill.exists():
        skill_text = handoff_skill.read_text(encoding="utf-8")

        # 3. SKILL.md saves to .handoffs/ with timestamps
        ok = ".handoffs/" in skill_text and re.search(r"\d{8}-\d{6}", skill_text) is not None
        msg = (
            "agent-handoff/SKILL.md: references .handoffs/ with timestamp pattern"
            if ok
            else "agent-handoff/SKILL.md: MISSING .handoffs/ reference or timestamp pattern"
        )
        details.append({"check_item": "handoff-storage", "passed": ok, "evidence": msg})
        all_pass = all_pass and ok

    # 4. AGENTS.md has Session Rituals or Handoff Protocol section
    ok, msg = _check_contains(agents_md, r"## (Session Rituals|Handoff Protocol)", "AGENTS.md session rituals / handoff section")
    details.append({"check_item": "agents-handoff-section", "passed": ok, "evidence": msg})
    all_pass = all_pass and ok

    # 5. AGENTS.md mentions compaction relationship
    ok, msg = _check_contains(agents_md, r"compaction", "AGENTS.md compaction mention")
    details.append({"check_item": "agents-compaction", "passed": ok, "evidence": msg})
    all_pass = all_pass and ok

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check handoff compliance")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache", help="Path to cache.json (optional)")
    parser.add_argument("--update-cache", action="store_true", help="Write result to cache")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "handoff-compliance"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["handoff-compliance"] = {
            "passed": result["passed"],
            "timestamp": __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S"),
            "ttl_s": 300,
            "details": result["details"],
        }
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
