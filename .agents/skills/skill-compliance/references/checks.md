# Check Registry

Each audit check is a standalone Python script under `scripts/`. All take
identical CLI arguments and return JSON to stdout.

| Check ID | Script | TTL | Verifies |
|----------|--------|-----|----------|
| `handoff-compliance` | `check_handoff_compliance.py` | 5 min | agent-handoff SKILL.md exists, saves to `.handoffs/`, AGENTS.md has protocol section with compaction mention |
| `structure` | `check_structure.py` | 1 h | Every skill dir has SKILL.md, catalog matches disk (no untracked or missing skills) |
| `invocation` | `check_invocation.py` | 1 h | `disable-model-invocation` frontmatter matches AGENTS.md invocation column |
| `completion-criteria` | `check_completion_criteria.py` | 1 h | Every step/phase section has a completion criterion |
| `description` | `check_description.py` | 1 h | Frontmatter `description` exists, one line, has triggers, no duplicates |
| `leading-words` | `check_leading_words.py` | 2 h | At least one leading word or core concepts section defined |

## Common CLI interface

```bash
python scripts/check_<name>.py \
    --skills-dir /path/to/.agents/skills \
    --cache /path/to/assets/cache.json \
    --update-cache
```

Returned JSON structure:

```json
{
  "check": "handoff-compliance",
  "passed": true,
  "details": [
    {"check_item": "handoffs-dir-exists", "passed": true, "evidence": ".handoffs/: found at ..."},
    ...
  ]
}
```
