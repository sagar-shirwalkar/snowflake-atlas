#!/usr/bin/env python3
"""Check skill directory structure against AGENTS.md catalog.

Verifies:
  - Every skill directory under .agents/skills/ has a SKILL.md
  - Every skill listed in AGENTS.md catalog has a matching directory
  - Every skill directory on disk is listed in AGENTS.md catalog
  - Reports untracked (new) and missing (deleted) skills
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _parse_agents_catalog(agents_md: Path) -> list[dict]:
    """Parse the AGENTS.md skill catalog table into a list of dicts.

    Scoped to only the ## Skill Catalog section to avoid matching
    other tables (MCP Server Entry Points, etc.).
    """
    if not agents_md.exists():
        return []

    text = agents_md.read_text(encoding="utf-8")
    skills = []

    # Find the ## Skill Catalog section boundaries
    catalog_start = re.search(r"^## Skill Catalog\s*$", text, re.MULTILINE)
    if not catalog_start:
        return []

    # Find the next ## heading after catalog start (end of section)
    next_section = re.search(
        r"^## \S", text[catalog_start.end():], re.MULTILINE
    )
    section_text = text[catalog_start.end():]
    if next_section:
        section_text = section_text[:next_section.start()]

    # Parse table rows within the section
    # Pattern: | `skill-name` | invocation | purpose |
    table_pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*(.+?)\s*\|",
        re.MULTILINE,
    )
    for match in table_pattern.finditer(section_text):
        name = match.group(1)
        invocation = match.group(2)
        purpose = match.group(3)
        if name in ("Skill", "---") or invocation in ("Invocation",):
            continue
        skills.append({"name": name, "invocation": invocation, "purpose": purpose})

    return skills


def run(skills_dir: str) -> dict:
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        return {"passed": False, "details": [{"check_item": "skills-dir-exists", "passed": False, "evidence": f"Skills directory not found: {skills_dir}"}]}

    # Discover skill dirs on disk (exclude hidden dirs, non-dirs)
    disk_skills = set()
    for entry in skills_path.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            disk_skills.add(entry.name)

    details = []
    all_pass = True

    # Check each disk skill has SKILL.md
    for name in sorted(disk_skills):
        skill_md = skills_path / name / "SKILL.md"
        ok = skill_md.exists()
        details.append({
            "check_item": f"skill-file-{name}",
            "passed": ok,
            "evidence": f"{name}/SKILL.md: {'found' if ok else 'MISSING'}",
        })
        all_pass = all_pass and ok

    # Parse AGENTS.md catalog
    agents_md = skills_path.parent.parent / "AGENTS.md"
    catalog_skills = _parse_agents_catalog(agents_md)
    catalog_names = {s["name"] for s in catalog_skills}

    # Skills on disk but not in catalog (untracked / new)
    untracked = disk_skills - catalog_names
    if untracked:
        all_pass = False
        details.append({
            "check_item": "untracked-skills",
            "passed": False,
            "evidence": f"Skills on disk but NOT in AGENTS.md catalog: {sorted(untracked)}",
        })
    else:
        details.append({
            "check_item": "untracked-skills",
            "passed": True,
            "evidence": "All skill directories are listed in AGENTS.md catalog",
        })

    # Skills in catalog but missing from disk (deleted)
    missing = catalog_names - disk_skills
    if missing:
        all_pass = False
        details.append({
            "check_item": "missing-skills",
            "passed": False,
            "evidence": f"Skills in AGENTS.md catalog but MISSING from disk: {sorted(missing)}",
        })
    else:
        details.append({
            "check_item": "missing-skills",
            "passed": True,
            "evidence": "All cataloged skills have directories on disk",
        })

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check skill structure compliance")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "structure"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["structure"] = {
            "passed": result["passed"],
            "timestamp": __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S"),
            "ttl_s": 3600,
            "details": result["details"],
        }
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
