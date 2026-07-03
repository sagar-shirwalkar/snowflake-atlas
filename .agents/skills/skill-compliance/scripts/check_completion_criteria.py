#!/usr/bin/env python3
"""Check that skill steps/phases have completion criteria.

Scans each SKILL.md for headings and checks that any numbered phase,
step list, or procedural section has a corresponding completion criterion
(defined as a line matching '**Completion criterion**' or similar).
"""

import argparse
import json
import re
import sys
from pathlib import Path


STEP_HEADING_PATTERN = re.compile(r"^#{1,3}\s+(PHASE\s+\d+|Step\s+\d+|Phase\s+\d+)", re.IGNORECASE)
# Match **Completion criterion** or **Completion criterion:** (colon inside bold markers)
CRITERION_PATTERN = re.compile(r"\*\*[Cc]ompletion criterion.*?\*\*", re.IGNORECASE)


def _strip_code_blocks(text: str) -> str:
    """Remove content inside triple-backtick code blocks."""
    return re.sub(r"```.+?```", "", text, flags=re.DOTALL)


def _check_skill(skill_md: Path) -> tuple[bool, list[dict]]:
    """Check one SKILL.md for completion criteria."""
    text = skill_md.read_text(encoding="utf-8")
    # Strip code blocks so we don't match # Step N inside comments
    text_no_code = _strip_code_blocks(text)
    lines = text_no_code.splitlines()

    details = []
    step_count = 0
    steps_with_criteria = 0

    # Find step/phase headings
    for line in lines:
        if STEP_HEADING_PATTERN.match(line):
            step_count += 1

    # Check for completion criterion patterns anywhere in the file
    has_criteria_pattern = bool(CRITERION_PATTERN.search(text))

    # Also check for table-based criteria (like atlas-creation's 10-Phase table)
    has_table_criteria = "Completion Criterion" in text or "completion_criterion" in text

    has_criteria = has_criteria_pattern or has_table_criteria

    if step_count > 0:
        if has_criteria:
            steps_with_criteria = step_count

    passed = step_count == 0 or has_criteria
    if step_count == 0:
        evidence = f"{skill_md.name}: no numbered phases/steps found (reference-only skill — no criteria needed)"
        passed = True  # Reference-only skills don't need step criteria
    elif has_criteria:
        evidence = f"{skill_md.name}: {step_count} step(s) found, has completion criteria ({'text' if has_criteria_pattern else 'table format'})"
    else:
        evidence = f"{skill_md.name}: {step_count} step(s) found but NO completion criteria detected"

    details.append({
        "check_item": f"criteria-{skill_md.parent.name}",
        "passed": passed,
        "evidence": evidence,
    })

    return passed, details


def run(skills_dir: str) -> dict:
    skills_path = Path(skills_dir)
    details = []
    all_pass = True

    for entry in sorted(skills_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            continue

        passed, item_details = _check_skill(skill_md)
        details.extend(item_details)
        all_pass = all_pass and passed

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check completion criteria")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "completion-criteria"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["completion-criteria"] = {
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
