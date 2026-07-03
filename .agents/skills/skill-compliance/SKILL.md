---
name: skill-compliance
description: Audit skill compliance across 6 dimensions. Use when the user asks to "check compliance", "audit skills", "run compliance checks", or "verify skill quality". Branches: audit-all (run all checks in order), audit (run specific check by name), fix (auto-fix a failing check).
disable-model-invocation: false
---

Audit the project's skills against a set of compliance checks. Each check is a standalone Python script in `scripts/` that follows a plugin pattern.

## Leading words

- **Audit** — A scripted compliance check with a binary outcome (pass/fail) and evidence. Each audit is independent, composable, and cacheable.
- **Plugin** — A script in `scripts/` that can be dropped in or removed without changing anything else. Adding a new check means adding one file and one entry in `references/checks.md`.
- **Staleness** — The risk that cached results no longer reflect reality. Managed by per-check TTLs and the agent always printing results after running.

---

## Branches

### audit-all

Run every check in order, reading from cache for non-stale checks.

**Completion criterion:** every check has a pass/fail result and the summary has been printed to chat.

1. **Resolve paths.** Determine the skills directory (`.agents/skills/`), the cache file (`assets/cache.json`), and the project root.
2. **Read cache.** Load `assets/cache.json`. If it doesn't exist or is empty, start fresh.
3. **Determine stale checks.** For each check in the registry (`references/checks.md`), compare the cached timestamp against the check's TTL. A check is stale if:
   - It has no cache entry, OR
   - `now - cached_timestamp > ttl`
4. **Run stale checks.** For each stale check, execute its script:
   ```bash
   python scripts/check_<name>.py \
       --skills-dir <path> \
       --cache <cache_path> \
       --update-cache
   ```
   The script writes its result into the cache and returns JSON to stdout.
5. **Print summary.** After all checks complete, print a concise summary to the chat with one line per check:

   ```
   ✓ handoff-compliance   — agent-handoff exists, AGENTS.md has protocol + compaction
   ✗ structure            — untracked skill: skill-compliance not in AGENTS.md catalog
   ✓ invocation           — all frontmatter/AGENTS.md consistent
   ...
   ```

   Lead with failures (✗) so the user sees problems first.
6. **Offer to fix.** For any failing check, ask if the user wants automatic remediation before ending.

### audit

Run a single check by name (e.g., `audit leading-words`, `audit handoff-compliance`). Same procedure as audit-all but skips the staleness check — always runs fresh.

**Completion criterion:** the named check has been run and its result printed to chat.

### fix

Auto-fix a failing check. Currently supported fixable checks:

- `structure` — Can create SKILL.md stubs for missing skill dirs, and can add missing entries to AGENTS.md catalog
- `invocation` — Can update AGENTS.md catalog to match frontmatter
- `handoff-compliance` — Can create `.handoffs/` directory if missing

**Completion criterion:** the fix has been applied and re-check passes.

---

## Caching

Results are stored in `assets/cache.json`. The TTL for each check is defined in `references/checks.md`. The cache avoids re-running expensive checks on every invocation while staleness bounds ensure results don't drift.

**Tradeoff:** A long TTL means results may be stale when skills have been edited. A short TTL means more runs. The defaults balance this: structural checks (structure, invocation) have 1-hour TTLs; volatile checks (handoff-compliance while actively editing) have 5-minute TTLs. Leading words rarely change, so 2 hours.

The agent always prints results to chat after running, so even if the cache says "passing", the user sees fresh output when a check was actually executed.

---

## Quick Reference

```bash
# Run all checks (uses cache for non-stale)
cd .agents/skills/skill-compliance

# Run a single check (always fresh)
python scripts/check_handoff_compliance.py --skills-dir ../

# Run and update cache
python scripts/check_structure.py --skills-dir ../ --cache assets/cache.json --update-cache
```

## Reference Files

| File | Contents |
|------|----------|
| `references/checks.md` | Registry of all check IDs, scripts, TTLs, and their CLI interface |
