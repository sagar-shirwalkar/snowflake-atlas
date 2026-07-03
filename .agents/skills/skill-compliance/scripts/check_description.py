#!/usr/bin/env python3
"""Check skill descriptions for hygiene.

Verifies:
  - Frontmatter `description` field exists
  - Description fits on one line (no line breaks)
  - Description provides triggers ("Use when", "Mention", etc.)
  - No trigger duplication within a single description
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _parse_frontmatter(text: str) -> dict:
    meta = {}
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return meta
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def _extract_triggers(description: str) -> list[str]:
    """Extract trigger phrases from a description."""
    triggers = []
    for m in re.finditer(r'"([^"]+)"', description):
        triggers.append(m.group(1))
    for m in re.finditer(r'(?:Use when|mentions?|asks? for|wants)\s+([^,;.]+)', description, re.IGNORECASE):
        triggers.append(m.group(1).strip())
    return triggers


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

        name = entry.name
        text = skill_md.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(text)
        description = frontmatter.get("description", "")
        is_user_invoked = frontmatter.get("disable-model-invocation", "false").lower() == "true"

        sub_fails = []

        # 1. Description exists
        if not description:
            details.append({
                "check_item": f"desc-exists-{name}",
                "passed": False,
                "evidence": f"{name}: NO description in frontmatter",
            })
            all_pass = False
            continue

        # 2. Fits on one line
        if "\n" in description:
            sub_fails.append("multiline description")
        one_line_ok = "\n" not in description

        # User-invoked skills don't need trigger phrases (human-facing description)
        if is_user_invoked:
            item_ok = one_line_ok
            issues = "; ".join(sub_fails) if sub_fails else ""
            evidence = (
                f"{name}: user-invoked — description OK (1 line)" if item_ok
                else f"{name}: user-invoked — description issues \u2014 {issues}"
            )
            details.append({
                "check_item": f"desc-{name}",
                "passed": item_ok,
                "evidence": evidence,
            })
            all_pass = all_pass and item_ok
            continue

        # 3. Has trigger phrases (model-invoked only)
        triggers = _extract_triggers(description)
        has_triggers = len(triggers) > 0

        # 4. No trigger duplication (same trigger mentioned twice)
        trigger_set = set(t.lower() for t in triggers)
        dupes = len(triggers) - len(trigger_set)
        no_dupes = dupes == 0

        if sub_fails:
            issues = "; ".join(sub_fails)
            evidence = f"{name}: description issues \u2014 {issues}"
            item_ok = False
        elif not has_triggers:
            evidence = f"{name}: description has no trigger phrases ('Use when', quotes, etc.)"
            item_ok = False
        elif not no_dupes:
            evidence = f"{name}: description has {dupes} duplicated trigger(s)"
            item_ok = False
        else:
            evidence = f"{name}: description OK (1 line, {len(trigger_set)} unique trigger(s))"
            item_ok = True

        details.append({
            "check_item": f"desc-{name}",
            "passed": item_ok,
            "evidence": evidence,
        })
        all_pass = all_pass and item_ok

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check description compliance")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "description"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["description"] = {
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
