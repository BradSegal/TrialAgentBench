# Work through a clinical research episode

An analyst estimates a kidney-treatment effect and investigates candidate arthritis treatments:
reading clinical records, running an analysis, commissioning a study and responding to its
findings. All names are fictional. A fixed script drives the same tools an agent would use, so
no model API key is needed and the run does not measure AI performance.

Use Python 3.11 or 3.12, uv and Docker, from a clone of the public harness repository:

```bash
uv sync --locked
docker build --tag trialagentbench/executor:1.0.1 executor
uv run trialagentbench demo run --output results/teaching-run
```

Open `results/teaching-run/index.html`. In the extracted toolkit, use its wheel-install
instructions and `execution` in place of `executor`. Recover an interrupted run with:

```bash
uv run trialagentbench demo run --output results/teaching-run --resume
```

To connect your own agent, copy and run the external adapter:

```bash
uv run python examples/clinical/run_external_provider.py --output results/external-teaching-run
```

Replace the fixed policy in `ExampleProvider.generate_turn` with your client, returning
`LLMResponse` and `ToolCall` objects. Use the campaign commands in
[RUNNING.md](../../docs/RUNNING.md) for benchmark tasks.
