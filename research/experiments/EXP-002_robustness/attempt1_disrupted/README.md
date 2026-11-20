# EXP-002 attempt 1: disrupted, kept for transparency

The first full execution of EXP-002 (2026-11-19, 16:56–17:16 UTC) ran while other work on the
same machine (CPU-only BERTScore / sentence-transformers verification and test suites) caused
memory pressure. `systemd-oomd` killed the Ollama server twice: at 17:09:24 UTC, during the
`paraphrase` arm, and at 17:15:25 UTC, at the end of the `freeform` arm. See
`journal_oom_excerpt.log`; memory and swap were above 90%.

## Consequences

- **Paraphrase arm:** 2 of 60 generations failed with connection errors. The platform recorded
  them as execution errors, not as wrong answers.
- **Latency:** latency is not comparable across arms. Requests after each restart include model
  reloading, and the machine was swapping.
- **Server state:** an arm whose server restarted mid-run does not run under the same server state
  as the others.

## Decision (made before looking at attempt-1 results in detail)

The whole study is repeated on a quiet machine with the same code and definitions, and attempt 2 is
the primary result.
- These files are the attempt-1 runs (`runs.tar.gz`) and the output of the pre-registered analysis
  applied to them (`results_tables.md`, `results.json`, `per_case.csv`).
- They are reported alongside attempt 2 as a replication under degraded conditions, not discarded.
