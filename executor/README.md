# TrialAgentBench analysis executor

The executor contains Python 3.12 and R 4.5 in one Docker environment. Python packages are
hash-locked in `requirements.lock`; direct R packages are version-pinned in the Dockerfile.
Runs record the image ID and complete Debian, Python and R inventories. The supported build
environment is Linux x86_64 with Docker.

## Build and attest

From the public checkout root, build the image used by the clinical demonstration:

```bash
docker build --iidfile executor-image.id --tag trialagentbench/executor:1.0.1 executor
export TRIALAGENTBENCH_EXECUTOR_IMAGE=trialagentbench/executor:1.0.1
```

In the offline toolkit the build context is `execution/`. The first build requires the pinned
base image and package sources. Campaign preparation authenticates its toolkit and probes the
image against the release manifest. Its `environment_sha256` binds installed inventories,
support files and execution limits; `context_sha256` binds the build context. The manifest's
image reference is a local tag, not the environment identity.

A campaign pins one local image ID for run, resume and replay. Rebuilt image IDs can differ
while environment inventories agree. Every environment probe must match the release. A new
preparation may accept an equivalent image; resuming a campaign still requires its recorded
local image ID.

Each container is limited to 2 CPUs, 4,096 MB and 300 seconds per execution, with no network.
A concurrent task reserves two container slots. Size CPU and memory for that concurrency and,
where applicable, the Docker VM. Docker images, build cache and installed host dependencies
are additional to the benchmark's storage estimates.

## Libraries and tables

Python includes pandas, PyArrow, SciPy, statsmodels, lifelines, scikit-learn, matplotlib and
jsonschema. R includes recommended packages, jsonlite, survey, sandwich and
`trialagentbenchR`. Read Parquet in R with:

```r
library(trialagentbenchR)
data <- read_trialagentbench_parquet("data/analysis.parquet")
```

The bridge uses pinned PyArrow and a bounded typed CSV. It preserves Boolean, numeric and
string values, leading zeros, literal `NA` and nulls. Nested, date/time and binary columns
fail explicitly. Integers outside R's exact numeric range must be released as strings. Neither
language can install packages during a task.

The vendored autograd-gamma wheel and source archive retain their MIT notices. Code is
licensed under Apache-2.0. Dataset contents and documentation are licensed under CC BY 4.0.
See [DATA_LICENSE](../DATA_LICENSE).
