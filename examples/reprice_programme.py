"""Reprice a finished TrialDev programme with the verifier, optionally scaling the profit per benefit unit."""

import argparse
import json
import tempfile
from pathlib import Path

from trialagentbench_verifier.resource_accounting import reconstruct_trialdev_terminal_value

from trialagentbench.contracts.release.archive_safety import materialized_release_archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--ledger", type=Path, required=True, help="The finished run's budget.json")
parser.add_argument("--profit-multiplier", type=float, default=1.0)
args = parser.parse_args()
catalog = Path("benchmark/full/trialdev-evaluator.assets.json")
sums = Path("benchmark/full/SHA256SUMS").read_text().splitlines()
digest = next(line.split()[0] for line in sums if line.endswith(catalog.name))
with materialized_release_archive(catalog, expected_sha256=digest, task_ids=(args.task,)) as release:
    case = json.loads((release.root / "participant/tasks" / args.task / "task.json").read_text())["case_id"]
    pool = release.root / "evaluation/worlds" / case / "pool/manifest.json"
    configuration = json.loads((release.root / "evaluation/tasks" / args.task / "authored_programme.json").read_text())
    configuration["preferences"]["annual_profit_per_benefit_unit_usd"] *= args.profit_multiplier
    with tempfile.NamedTemporaryFile("w", suffix=".json") as changed:
        print(json.dumps(configuration), file=changed, flush=True)
        paths = {"ledger_path": args.ledger, "configuration_path": Path(changed.name), "pool_path": pool}
        value = reconstruct_trialdev_terminal_value(**paths)
print(f"{args.task}: development value US${value.net_value_usd / 1e6:.1f}M at {args.profit_multiplier}x profit")
