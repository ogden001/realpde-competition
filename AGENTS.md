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
