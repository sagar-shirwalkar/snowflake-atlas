---
name: security-review
description: Scan codebases for security vulnerabilities — injection flaws, auth bypass, secrets exposure, weak crypto, insecure deps, and business logic bugs. Use when asked to review code for security issues, audit for vulnerabilities, check for SQLi/XSS/command injection, find exposed API keys or hardcoded secrets, or any request like "is my code secure?", "security audit", or "check for vulnerabilities". Covers Python, TypeScript, JavaScript, Java, Go, Rust, Ruby, PHP.
disable-model-invocation: false
---

An AI-powered security scanner that reasons about your codebase the way a human security researcher would — tracing data flows, understanding component interactions, and catching vulnerabilities that pattern-matching tools miss.

## Leading words

- **Scope** — Determine the attack surface: which files, languages, frameworks, and entry points are in play. A scoped review of `src/auth/` is sharper than a whole-repo skim.
- **Audit** — Check dependencies (known CVEs), scan for hardcoded secrets (API keys, tokens, credentials), then deep-scan for injection, auth, crypto, and business logic flaws. Each layer catches things the others miss.
- **Trace** — Follow user-controlled input from entry points (HTTP params, headers, file uploads) all the way to sinks (DB queries, exec calls, HTML output). The most dangerous bugs span multiple files.
- **Verify** — Self-check every finding: is it actually exploitable, or is there sanitization you missed? Downgrade or discard false positives before reporting.

## Phases

### PHASE 1: Scope & dependency audit

**Completion criterion:** project languages and frameworks identified, dependencies checked for known CVEs, and the dependency audit logged.

1. Identify the language(s) and framework(s) in use (check `pyproject.toml`, `requirements.txt`, `package.json`, `go.mod`, `Cargo.toml`, `pom.xml`, `Gemfile`).
2. Read `references/language-patterns.md` to load framework-specific vulnerability patterns.
3. Audit dependencies for known vulnerable packages. Read `references/vulnerable-packages.md` for the curated watchlist.
4. Flag packages with known CVEs, deprecated crypto libs, or suspiciously old pinned versions.

### PHASE 2: Secrets & exposure scan

**Completion criterion:** all files scanned for hardcoded secrets, credentials, and sensitive data exposure. Findings logged with file paths.

1. Scan ALL files (including config, `.env`, CI/CD, Dockerfiles, IaC) for hardcoded API keys, tokens, passwords, private keys.
2. Check for committed `.env` files, secrets in comments or debug logs, cloud credentials (AWS, GCP, Azure, Stripe, etc.).
3. Read `references/secret-patterns.md` for regex patterns and entropy heuristics.

### PHASE 3: Vulnerability deep scan

**Completion criterion:** injection, auth, data handling, crypto, and business logic flaws all checked. Findings logged per category.

Read `references/vuln-categories.md` for full detection guidance. Cover these categories:

- **Injection flaws** — SQLi, XSS, command injection, LDAP/XPath/Header injection
- **Authentication & access control** — missing auth, BOLA/IDOR, JWT weaknesses, CSRF, privilege escalation, mass assignment
- **Data handling** — sensitive data in logs/errors, missing encryption, insecure deserialization, path traversal, XXE, SSRF
- **Cryptography** — weak hashes (MD5, SHA1), hardcoded IVs/salts, weak RNG, missing TLS validation
- **Business logic** — race conditions (TOCTOU), integer overflow, missing rate limiting, predictable resource IDs

### PHASE 4: Cross-file data flow analysis

**Completion criterion:** user-controlled input traced from entry points to sinks across file boundaries. Cross-file vulnerabilities identified.

1. Trace user-controlled input from entry points (HTTP params, headers, body, file uploads) all the way to sinks (DB queries, exec calls, HTML output, file writes).
2. Identify vulnerabilities that only appear when looking at multiple files together.
3. Check for insecure trust boundaries between services or modules.

### PHASE 5: Self-verify & report

**Completion criterion:** every finding self-verified, severity assigned, and both the full report and targeted patches are ready for human review.

For EACH finding:
1. Re-read the relevant code with fresh eyes. Is this actually exploitable, or is there sanitization you missed?
2. Check if a framework or middleware already handles this upstream.
3. Downgrade or discard findings that aren't genuine vulnerabilities.
4. Assign final severity: CRITICAL / HIGH / MEDIUM / LOW / INFO.

Generate the full report using the format in `references/report-format.md`.

### PHASE 6: Propose patches

**Completion criterion:** concrete patches proposed for every CRITICAL and HIGH finding. Nothing auto-applied.

For every CRITICAL and HIGH finding:
- Show the vulnerable code (before)
- Show the fixed code (after)
- Explain what changed and why
- Preserve the original code style, variable names, and structure
- Add a comment explaining the fix inline

Explicitly state: **"Review each patch before applying. Nothing has been changed yet."**

## Severity Guide

| Severity | Meaning | Example |
|----------|---------|---------|
| 🔴 CRITICAL | Immediate exploitation risk, data breach likely | SQLi, RCE, auth bypass |
| 🟠 HIGH | Serious vulnerability, exploit path exists | XSS, IDOR, hardcoded secrets |
| 🟡 MEDIUM | Exploitable with conditions or chaining | CSRF, open redirect, weak crypto |
| 🔵 LOW | Best practice violation, low direct risk | Verbose errors, missing headers |
| ⚪ INFO | Observation worth noting, not a vulnerability | Outdated dependency (no CVE) |

## Output Rules

- **Always** produce a findings summary table first (counts by severity)
- **Never** auto-apply any patch — present patches for human review only
- **Always** include a confidence rating per finding (High / Medium / Low)
- **Group findings** by category, not by file
- **Be specific** — include file path, line number, and the exact vulnerable code snippet
- **Explain the risk** in plain English — what could an attacker do with this?
- If the codebase is clean, say so clearly: "No vulnerabilities found" with what was scanned

## Reference Files

| File | Contents |
|------|----------|
| `references/vuln-categories.md` | Deep reference for every vulnerability category with detection signals, safe patterns, and escalation checkers |
| `references/secret-patterns.md` | Regex patterns, entropy-based detection, and CI/CD secret risks |
| `references/language-patterns.md` | Framework-specific vulnerability patterns for JS, Python, Java, PHP, Go, Ruby, Rust |
| `references/vulnerable-packages.md` | Curated CVE watchlist for npm, pip, Maven, Rubygems, Cargo, Go modules |
| `references/report-format.md` | Structured output template for security reports with finding cards, dependency audit, secrets scan, and patch proposals |
