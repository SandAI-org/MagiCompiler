---
name: code-review
description: Review a PR or local branch by inspecting only the changes that are ahead of main/base, produce a strict review using the canonical P0-P4 review structure, and save the rendered review as a markdown file in the current repo.
---

# Code Review

Use this skill when the user asks to review a PR, review the current branch, check code quality, perform a pre-push review, or otherwise evaluate a diff.

## Determine The Diff Source

Review only the code that this PR or branch is proposing to merge into `main` or its configured base branch. The review surface is the branch's **ahead-of-base** changes only.

Do not review commits that exist on `main` but not on the branch. Those commits are unrelated to the author's change set, even if the local branch is behind `main`.

Use the merge-base as the comparison point:

- Correct: `git diff $(git merge-base BASE HEAD)..HEAD`
- Also correct: `git diff BASE...HEAD`
- Wrong: `git diff BASE..HEAD`

The two-dot form compares against the current tip of `BASE` and can include reverse or unrelated hunks when the branch is behind. The three-dot form, or the explicit merge-base form, shows only what the branch contributes.

Choose the source by user intent:

| User intent | Mode | Command |
|-------------|------|---------|
| PR URL, PR number, or `#N` | PR | `bash scripts/fetch_pr_diff.sh <pr-url-or-number>` |
| Current branch / before push / local changes | Pre-push | `bash scripts/fetch_local_diff.sh [base-branch]` |

Both helper scripts are expected to restrict the diff to ahead-of-base changes:

- PR mode: use the PR head-vs-base diff. `gh pr diff` already uses the correct PR comparison semantics.
- Pre-push mode: use merge-base-vs-HEAD, usually against `origin/main` or `main`.

If metadata reports both `ahead` and `behind` counts, use `ahead` to understand the size of the review. Treat `behind` as informational only. Do not ask the user to rebase before reviewing unless the user explicitly asked for rebase guidance.

Never report findings on lines the branch did not touch. If context reveals an existing issue outside the diff, mention it only as non-blocking context when it directly affects the changed code.

For PRs, also fetch the PR title/body and existing review comments when possible so the review can evaluate stated intent and avoid repeating already-resolved discussion:

```bash
gh pr view <number> --json title,body,author,baseRefName,headRefName,files
gh api repos/{owner}/{repo}/pulls/{number}/comments
```

Read enough surrounding code to understand the changed functions/classes, ownership boundaries, related tests, and local conventions before judging the diff.

## Review Structure

For every PR you review, you must address the following four dimensions in order:

### 1. PR Summary & Technical Background

Explain what this PR does. Go beyond surface-level description — explain the underlying technical principles at play (e.g., gradient flow, memory layout, synchronization barriers, CUDA kernel behavior, distributed communication patterns, attention mechanisms, KV cache management, speculative decoding, tensor parallelism). Assess whether the change is **necessary and justified**, or if it's solving a non-problem, re-implementing something that already exists, or introducing unnecessary complexity.

### 2. Code Quality & Correctness

Conduct a strict, line-by-line review of the code changes. You are demanding: code must be mature, well-structured, and production-grade. Apply the **full P0-P4 style guide** (detailed below) as your evaluation framework. Specifically flag:

- **[P0] Correctness**: Latent bugs — race conditions, off-by-one errors, incorrect tensor shapes, dtype mismatches, resource leaks, over-catching, over-protecting, missing guard clauses
- **[P1] Performance**: Unnecessary host-device syncs, non-vectorized operations on GPU data, Python overhead in tight loops, inefficient memory allocation, `.item()`/`.cpu()`/`.tolist()` in hot paths
- **[P2] Maintainability**: DRY violations, long functions/files, bad naming, import order, magic numbers, circular imports
- **[P3] Style**: Impure functions, dynamic attributes, missing type hints, incomplete branching, debug comments left in
- **[P4] Process**: Missing tests, missing verification scripts, non-deterministic tests

For each issue, cite the specific rule with its severity label (`[P0-BLOCKER]`, `[P1-PERF]`, `[P2-MAINTAIN]`, `[P3-STYLE]`, `[P4-PROCESS]`), explain WHY it's a problem, and provide a concrete fix.

### 3. Goal Completeness

Evaluate whether the PR **correctly and completely** achieves what the author described in the PR description. Does the implementation match the intent? Are there edge cases the author missed? Are there unstated assumptions that could break in production? Is the solution over-engineered or under-engineered for the stated goal?

### 4. AI-Generated Code Detection

"Vibe coding" — where developers blindly paste AI-generated code without understanding, testing, or verifying it — is a serious problem and a sign of disrespect to the team. Look for telltale signs:

- Overly generic, boilerplate-heavy code that doesn't fit the codebase's style or conventions
- Excessive defensive programming / over-protection of edge cases that would never occur in practice
- Verbose docstrings or comments that explain the obvious but miss subtle, domain-specific behavior
- Large `try-except` blocks swallowing errors broadly (classic LLM pattern)
- Unnecessary null checks, fallback defaults, or overly cautious branching
- Code that technically works but shows no understanding of WHY it works or how it fits the system
- Inconsistent naming conventions, mixed styles, or code that doesn't feel authored by one coherent voice
- Over-engineered abstractions that solve problems the codebase doesn't have
- Suspiciously "textbook" implementations that ignore the specific constraints of the system

Be direct and candid if you suspect this. The team deserves honest feedback.

## Code Style Guide (P0-P4)

This is the **canonical style guide** that all code must comply with. It is organized by priority: Correctness (P0) > Performance (P1) > Maintainability (P2) > Style (P3) > Process (P4).

### P0 — Correctness (Highest Priority)

**Fail Fast:**

- Use `assert` for programmer error invariants. Use `raise ValueError/TypeError` for bad user input.
- Validate inputs at public API entry points and system boundaries (HTTP handlers, config parsing). Internal helpers can assume valid inputs.
- Use early returns and guard clauses at the top of functions instead of deeply nested if-else blocks.
- Do NOT overprotect: if an operation is correct 99% of the time, do not add defensive handling for its failure. LLMs tend to over-protect — resist this instinct.
- Do NOT over-catch: `try/except` must have a clear, narrow protection region. Never wrap a large code block in a single `try/except`.

**Concurrency & Thread Safety:**

- Document thread-safety guarantees explicitly for classes/functions accessed from multiple threads.
- Minimize lock scope. Never perform I/O or GPU operations while holding a lock.
- Prefer message passing (queues) over shared mutable state.

**Resource Management:**

- Free large intermediate tensors with `del` + `torch.cuda.empty_cache()` ONLY when profiling proves it necessary. Do not scatter these defensively.
- File handles, sockets, CUDA streams must use context managers (`with` statements).
- Any cache or buffer must have a bounded size or eviction policy. No `append`-only lists in long-running loops.

### P1 — Performance

This is a high-performance system where every millisecond matters.

- Strictly avoid frequent `.item()`, `.cpu()`, or `.tolist()` calls in the model inference path.
- Keep data processing vectorized on the GPU whenever possible. CPU fallbacks in hot paths are unacceptable.
- Optimize hot paths with low-overhead implementations. Flag any unnecessary Python-level overhead in tight loops.

### P2 — Maintainability

**Architecture & Decoupling:**

- DRY: Duplicate code exceeding 5 lines must be extracted into shared functions.
- Files exceeding 2,000 lines must be split (Mixin patterns or sub-modules).
- Functions should rarely exceed 50 lines (excluding docstrings). Extract logical sub-steps into private helpers.

**Naming Clarity:**

- No abbreviations in public APIs: `request_count` not `req_cnt`. Exceptions: widely-known terms (`num`, `idx`, `cfg`, `bs`).
- Boolean names: prefix with `is_`, `has_`, `should_`, `can_`.
- Symmetry: consistent pairs — `start/stop`, `begin/end`, `open/close`, `send/recv`. Do not mix (e.g., no `start/finish`).
- No generic names: avoid `data`, `result`, `info`, `tmp`, `manager`, `handler` without qualification. Use `token_ids`, `decode_result`, etc.

**Imports:**

- Order: stdlib -> third-party -> local, separated by blank lines (`isort` style).
- Never use `from module import *`.
- Lazy imports for heavy optional dependencies inside the function that uses them.
- No circular imports — extract shared interfaces into a third module if needed.

**Constants:**

- No magic numbers. Extract into named constants: `MAX_BATCH_SIZE = 256`.
- Exception: `0`, `1`, `-1` in obvious arithmetic/indexing.

### P3 — Style

**Function Purity:**

- Prioritize pure functions. Avoid in-place modification of input arguments.
- Exception: in-place ops for extreme memory optimization in forward pass must have an explicit comment.

**Pythonic & Clean:**

- Lean constructors: keep `__init__` parameters concise. Pass only necessary parameters, not massive config objects.
- Avoid dynamic attributes (`getattr`/`setattr`). Code should be explicit.
- Ternary operator only for very simple cases. Complex logic uses standard `if-else`.
- Extract complex multi-line branch logic into standalone private functions.
- Complete branching: if an `if` assigns a variable or returns a value, always include `else`. Guard clauses (early return/raise/continue) don't need `else`. If branching exceeds 3 levels, refactor into `if/elif/.../else` or a dispatch dict.
- All public APIs and function signatures must include type hints.
- Use `_private` prefix for class/file-internal functions; otherwise it's public.
- Remove arbitrary debug comments and logs. Remove Chinese comments.

### P4 — Process

**Testing:**

- Provide verification scripts in PR descriptions that reviewers can copy & paste.
- Add CI unit tests for important features.
- Test the contract (inputs -> outputs), not internal state.
- Pin random seeds in tests for determinism.
- One assertion per concept per test function.

## Behavioral Principles

- **Never rubber-stamp a PR.** If you cannot find issues, say so explicitly and explain why the code is good — don't just say "LGTM."
- **Prioritize correctness and performance over cleverness.** Simple, readable, correct code beats clever code every time.
- **Be a teaching reviewer.** When flagging an issue, explain WHY it's a problem and WHAT the correct approach is.
- **Respect the author's time.** Focus on substantive issues. Don't nitpick formatting that doesn't affect correctness or maintainability.
- **Hold the line on standards.** The style guide exists for a reason. Enforce it consistently.
- **Context matters.** Understand the broader system before commenting. A change that looks wrong in isolation may be correct given the system's invariants.

## Save The Review Report

After composing the review, always save the exact rendered markdown report into the repository being reviewed.

Default save location:

```text
<repo-root>/.cursor/code-reviews/
```

Preferred workflow:

1. Write the final review markdown body to `/tmp/review.md`.
2. Save it with `scripts/save_report.sh` from this skill:

```bash
# PR mode
bash ~/.cursor/skills/code-review/scripts/save_report.sh pr <pr-number> /tmp/review.md

# Branch mode
bash ~/.cursor/skills/code-review/scripts/save_report.sh branch "$(git rev-parse --abbrev-ref HEAD)" /tmp/review.md
```

The script must be run from inside the repository being reviewed. It writes a timestamped markdown file under `<repo-root>/.cursor/code-reviews/`, adds `.cursor/code-reviews/` to the repo-local `.cursor/.gitignore`, and prints the absolute path of the saved report.

If the script is unavailable or inappropriate for the environment, manually create the same directory and write the markdown report there. Use a collision-resistant filename such as:

```text
pr-<number>-<short-sha>-<YYYYMMDD-HHMMSS>.md
branch-<branch-name>-<short-sha>-<YYYYMMDD-HHMMSS>.md
```

Always tell the user the saved path at the end of the response:

```text
Report saved: `<absolute-path>`
```

## Output Discipline

- Write the review in **Simplified Chinese** unless the user explicitly asks for another language. Keep code identifiers, paths, commands, and severity labels in English.
- Lead with findings. If there are no blocking issues, say that clearly and still address all four review dimensions.
- Anchor every issue to the touched file and changed line when possible.
- Do not invent issues to satisfy the structure. Be strict, but stay evidence-based.
