#!/usr/bin/env python3
"""Check disable-model-invocation consistency between SKILL.md and AGENTS.md.

Verifies:
  - If disable-model-invocation: true → AGENTS.md marks as **user-invoked**
  - If disable-model-invocation: false (or absent) → AGENTS.md marks as model-invoked
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter as a dict (naive parser, no yaml dep needed)."""
    meta = {}
    # Match YAML frontmatter between --- delimiters
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return meta
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def _parse_agents_catalog(agents_md: Path) -> dict[str, str]:
    """Return {skill_name: invocation_type} from AGENTS.md catalog table.

    Scoped to the ## Skill Catalog section only.
    """
    if not agents_md.exists():
        return {}
    text = agents_md.read_text(encoding="utf-8")

    catalog_start = re.search(r"^## Skill Catalog\s*$", text, re.MULTILINE)
    if not catalog_start:
        return {}

    next_section = re.search(
        r"^## \S", text[catalog_start.end():], re.MULTILINE
    )
    section_text = text[catalog_start.end():]
    if next_section:
        section_text = section_text[:next_section.start()]

    catalog = {}
    table_pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|",
        re.MULTILINE,
    )
    for match in table_pattern.finditer(section_text):
        name = match.group(1)
        invocation = match.group(2)
        if name in ("Skill", "---") or invocation in ("Invocation",):
            continue
        catalog[name] = invocation
    return catalog


def run(skills_dir: str) -> dict:
    skills_path = Path(skills_dir)
    agents_md = skills_path.parent.parent / "AGENTS.md"
    catalog = _parse_agents_catalog(agents_md)

    details = []
    all_pass = True

    # Discover skill dirs
    disk_skills = []
    for entry in skills_path.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            disk_skills.append(entry.name)

    for name in sorted(disk_skills):
        skill_md = skills_path / name / "SKILL.md"
        if not skill_md.exists():
            continue

        frontmatter = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        disabled = frontmatter.get("disable-model-invocation", "false").lower() == "true"
        catalog_invocation = catalog.get(name, "")

        if disabled:
            expected = "**user-invoked**"
            ok = expected in catalog_invocation
            evidence = (
                f"{name}: disable-model-invocation=true → catalog='{catalog_invocation}' "
                f"{'✓ user-invoked' if ok else '✗ should be user-invoked'}"
            )
        else:
            expected = "model-invoked"
            ok = expected in catalog_invocation
            evidence = (
                f"{name}: disable-model-invocation=false → catalog='{catalog_invocation}' "
                f"{'✓ model-invoked' if ok else '✗ should be model-invoked'}"
            )

        details.append({
            "check_item": f"invocation-{name}",
            "passed": ok,
            "evidence": evidence,
        })
        all_pass = all_pass and ok

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check invocation compliance")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "invocation"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["invocation"] = {
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
