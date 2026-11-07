# Examples

These files support the [running guide](../docs/RUNNING.md).

- `smoke.json` is a three-task campaign run by scripted controls, with no model calls.
- `campaign.json` is a small campaign to copy and edit for your own model.
- `campaign.annotated.jsonc` lists every campaign setting with a comment; it is a reference, not a runnable file.
- `custom_agent.py` runs your own agent on one benchmark task, then reprices the finished programme with the
  verifier. Its agent is a local script, so it makes no model calls and needs no key; replace `generate_turn` with
  your client.
- `rule_based_developer.py` and `rule_readout.py` are the rule-based developer behind the TrialDev comparison.
- `reprice_programme.py` recomputes a finished programme's value with the verifier, optionally at a different profit.
- [`clinical/`](clinical/README.md) is a clinical walkthrough driven by a fixed script, with its two small archives.

The first command runs the demonstration, which needs the executor image and no download. The others read the release
downloaded to `benchmark/`.

```bash
uv run trialagentbench demo run --output results/demo
uv run trialagentbench campaign preview --config examples/smoke.json
uv run python examples/custom_agent.py --output results/custom-agent
uv run python examples/rule_based_developer.py --task td-87d6661cdfed --output results/rule-based
uv run python examples/reprice_programme.py --task td-87d6661cdfed --ledger results/rule-based/budget.json --profit-multiplier 2
```
