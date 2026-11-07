# TrialAgentBench verifier

The verifier authenticates release downloads and recomputes scores from recorded campaign
files. It implements the scorer independently and imports neither the harness nor the
simulator. The [scoring guide](../docs/METHODS.md) defines the metrics;
[RUNNING.md](../docs/RUNNING.md) explains campaigns. Every command lists its arguments with
`--help`.

## Install and check

From the public checkout, `uv sync --locked` installs the verifier with the harness. For a
standalone installation use `python -m pip install ./verifier` or the toolkit wheel. Omit
`uv run` for a pip installation.

```bash
uv run trialagentbench-verify --help
uv run trialagentbench-verify practice --output practice-report.json --review-dir practice-review
```

Practice checks packaged known-answer cases without Docker, credentials or a dataset download.
Commands that execute submitted analyses need Docker; re-scoring recorded outputs does not.

## Authenticate released inputs

Use expected digests retained from the pinned release you trust. Checksums detect corruption
and mismatches; obtaining them beside files does not establish their publisher.

| Command fragment | Scope |
| --- | --- |
| `release-set DIR --expected-manifest-sha256 D` | Complete manifest, catalogs, toolkit and every shard |
| `release CATALOG --expected-sha256 D [--task-id T ...]` | Selected logical bytes and supplied verification records |
| `toolkit ZIP --expected-sha256 D` | Toolkit members and distributions |
| `assets fetch CATALOG --expected-sha256 D --cache-directory C` | Verified selective acquisition |

## Re-score recorded work

Keep the whole campaign directory. The verifier rescores stored replay outcomes; it does not
rerun them.

| Command | Recomputed result |
| --- | --- |
| `analyse-trialeval` | Replay loss, decision error rates, accuracy, completion and paired comparisons |
| `analyse-trialdev` | One run's initial-analysis loss, acquired-study scores and development value |
| `compare-trialdev` | Whole-campaign paired comparisons over independent worlds |
| `execute` | Fresh reproduction of a saved analysis in the pinned executor |

### Reproduce truth recovery

Truth recovery is TrialEval's primary measure. This command aggregates per-answer rows and scores
nothing. `schedule.csv` has the columns `model,task_id` and lists every scheduled task.

```bash
uv run python -m trialagentbench_verifier.resources.truth_recovery \
  --answers comparisons/trialeval/trialeval_answers.csv --schedule schedule.csv \
  --evaluator-release EVALUATOR --replay-bank replay-bank \
  --replay-bank-sha256 "$REPLAY_BANK_SHA256" --output truth-recovery
```

`EVALUATOR` is an extracted TrialEval evaluator release. Rows from the public bank carry the
process-aware analysis only; add `--fixed-analyses benchmark/fixed-analyses` for the other fixed
analyses. For TrialDev, add `--trialdev-rankings`, `--oracle-values` and
`--expected-oracle-values-sha256`.

`policy:follow_plan` is Plan's primary analysis, `policy:always_robust` is Always corrected, `policy:always_naive` is
Never corrected, `policy:scripted_reference` is Process-aware analysis.

Code is licensed under Apache-2.0. Dataset contents and documentation are licensed under CC BY
4.0. Legal texts: [Apache-2.0](LICENSE), [CC BY 4.0](../DATA_LICENSE).
