# Track 1 Execution Rules

## Required context

Before planning, modifying code, or interpreting an experiment, read:

1. the workspace-root `MEMORY.md`;
2. this repository's `README.md` and relevant `docs/` material;
3. `docs/coordination/STATUS.md` first for the current handoff state; and
4. `docs/track1_experiment_registry.md` only to the extent relevant to the requested task.

`STATUS.md` is a short handoff index, not a replacement for the registry or `MEMORY.md`. `submission_log.md` records Codabench submissions only.

## ChatGPT–Codex coordination

Treat a user-provided `NEXT_ACTION` as the current bounded task. It must state, or Codex must establish from the existing protocol: goal, allowed data/resource budget, prohibited actions, and acceptance criteria. Do not expand scope.

For each material experiment or candidate, append verifiable facts to `docs/track1_experiment_registry.md`: experiment ID, state (`DONE`, `REVIEW_REQUIRED`, `BLOCKED`, or `RUNNING`), commit SHA or explicit dirty-tree note, split/manifest SHA, exact command and key configuration, artifact ID or repository-relative path, core metrics/observations, and a `GO`/`STOP`/`REVIEW_REQUIRED` conclusion. Do not overwrite prior conclusions.

On task completion, update `docs/coordination/STATUS.md` with the latest completed item, current state, allowed scope, and review handoff. Update stable registry conclusions only when evidence exists. Do not put datasets, checkpoints, credentials, absolute private paths, or generated submission archives in Git.

## Long-running work

Before a long CPU/GPU job, implement and smoke-test first; state the host, command, log and artifact locations, and a monitoring command. Start the runner detached, verify its PID/log once, record `RUNNING`, then stop rather than polling. Recover results only on a later explicit request.

Do not assume GitHub push/pull access or that an external ChatGPT session can read this private repository. Record local facts here; report any unpushed coordination changes to the user.

## Git Docs as long-term technical memory

Git `docs/` is the shared long-term technical memory for ChatGPT/Sol, Codex,
and the research team. Conversation context is short-lived and must not replace
stage conclusions recorded in Git.

The intended information flow is: experiment or analysis facts → concrete
experiment record → stable stage conclusion → the relevant direction overview
→ `docs/realpde整体优化概要.md`. Concrete records retain full configuration,
commands, provenance, and metrics; overviews remain short and link back to that
evidence.

ChatGPT/Sol owns technical direction, experiment design and result review,
including `KEEP` / `REVIEW` / `NO-GO` / `STOP` decisions and priority. After an
important stable conclusion, it should identify the conclusion and the overview
documents that need updating. Codex owns execution facts and artifacts, detailed
records, and incremental overview updates once a conclusion is confirmed. Update
the overall overview only when a first-level direction materially changes.

Do not update an overview for an in-progress run, an intermediate checkpoint, or
an unstable observation. Update it when evidence establishes a useful method, a
failed or stopped route worth not repeating, a key technical insight, a priority
change, a completed exploration phase, or a result that changes the next route.
Do not turn an overview into an experiment diary or copy a long report into it.
Superseded conclusions should be replaced; material `NO-GO` / `STOP` outcomes
should remain as concise anti-duplication memory.

Before starting work in an existing direction, read its overview first, then the
linked detailed evidence needed for the bounded task. Flag a proposed experiment
that repeats a documented `NO-GO` / `STOP` result or conflicts with an existing
conclusion rather than silently rerunning it.
