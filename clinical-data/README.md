# Clinical data tables

Deterministic projections into Study Data Tabulation Model (SDTM) and Analysis Data Model (ADaM)
tables used by TrialAgentBench.
These functions consume observed tables; they generate neither patients nor outcomes.
The verifier's scientific review independently checks the exported values and linkage.

`clinical_timing` converts elapsed observations and optional participant reference
instants into calendar dates and study days. The exposure, findings and analysis
projectors accept the same row-aligned `reference_days` input for staggered entry.
An unknown calendar origin remains unknown; it is not replaced with an invented date.

In the public TrialAgentBench checkout this package lives at `clinical-data/` and is installed
with the harness by `uv sync --locked` from the repository root. For a standalone installation
from that same root:

```bash
python -m pip install ./clinical-data
```

The import namespace is `clinical_data_tables`. This package has no command-line entrypoint;
benchmark execution and verification use `trialagentbench` and `trialagentbench-verify`.
