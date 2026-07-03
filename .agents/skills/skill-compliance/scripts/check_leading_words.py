#!/usr/bin/env python3
"""Check that skills define and use leading words.

A leading word is a compact concept anchored in the model's pretraining
that the skill uses to anchor behaviour.  Common indicators:
  - A "## Leading words" or "## Core concepts" section
  - Bolded key terms with explanations
  - The word **leading word** or **leading** appears in the text
"""

import argparse
import json
import re
import sys
from pathlib import Path

LEADING_SECTION_PATTERN = re.compile(
    r"^#{1,3}\s+(?:Core concepts|Leading words?|Leading word|Key concepts)",
    re.IGNORECASE | re.MULTILINE,
)

BOLD_TERM_PATTERN = re.compile(r"\*\*([^*]+)\*\*\s*—")


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

        # Skip skills that explicitly declare no leading words in frontmatter
        frontmatter = _parse_frontmatter(text)
        if frontmatter.get("leading-words", "").lower() == "none":
            details.append({
                "check_item": f"leading-{name}",
                "passed": True,
                "evidence": f"{name}: explicitly declares no leading words (skipped)",
            })
            continue

        # Check 1: Has a "Leading words" or "Core concepts" section
        has_section = bool(LEADING_SECTION_PATTERN.search(text))

        # Check 2: Has bolded terms with explanations (the "word — definition" pattern)
        bold_terms = BOLD_TERM_PATTERN.findall(text)
        has_bold_defs = len(bold_terms) >= 1

        passed = has_section or has_bold_defs

        if passed:
            sources = []
            if has_section:
                sources.append("leading-words section")
            if has_bold_defs:
                sources.append(f"{len(bold_terms)} bold-defined term(s)")

            # Find the actual leading words for evidence
            if has_section:
                # Extract the section content
                section_match = LEADING_SECTION_PATTERN.search(text)
                if section_match:
                    # Find the next heading after this one
                    rest = text[section_match.end():]
                    next_heading = re.search(r"^#{1,3}\s", rest, re.MULTILINE)
                    section_content = rest[:next_heading.start()] if next_heading else rest[:500]
                    words = BOLD_TERM_PATTERN.findall(section_content)
                    if words:
                        evidence = f"{name}: leading words: {', '.join(words)}"
                    else:
                        evidence = f"{name}: has {', '.join(sources)}"
                    all_pass = all_pass and True
                else:
                    evidence = f"{name}: has {', '.join(sources)}"
            else:
                evidence = f"{name}: {len(bold_terms)} bold-defined term(s) found"
        else:
            evidence = f"{name}: NO leading words or core concepts section found"
            all_pass = False

        details.append({
            "check_item": f"leading-{name}",
            "passed": passed,
            "evidence": evidence,
        })

    return {"passed": all_pass, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Check leading words compliance")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--update-cache", action="store_true")
    args = parser.parse_args()

    result = run(args.skills_dir)
    result["check"] = "leading-words"

    if args.cache and args.update_cache:
        cache_path = Path(args.cache)
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["leading-words"] = {
            "passed": result["passed"],
            "timestamp": __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S"),
            "ttl_s": 7200,
            "details": result["details"],
        }
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
