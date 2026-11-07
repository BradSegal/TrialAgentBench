"""Choose a small early-safety study from the participant's first reporting targets.

Sites open in the public activation order (documents/site_network.csv), at least two, until they list
480 candidates and forecast 60 screened before the last enrolment day that still reads out by day 365:
under registry-rate accrual a short site queue can open late, screen slowly, and hold fewer eligible
candidates than a study needs once screen failures are counted.
"""

import csv
import json
from pathlib import Path

CANDIDATES, SCREENED, LAST_ENROLMENT_DAY = 480, 60, 365 - 56

with Path("interface/targets.csv").open(newline="") as handle:
    compound = next(csv.DictReader(handle))["target_id"].split(":")[0]
with Path("documents/site_network.csv").open(newline="") as handle:
    network = sorted(csv.DictReader(handle), key=lambda site: int(site["activation_order"]))


def screened(site: dict[str, str]) -> float:
    """Candidates the site is forecast to screen between its opening and the last enrolment day."""
    months = max(0.0, LAST_ENROLMENT_DAY - float(site["opening_day"])) / 30.4375
    return float(site["screening_candidates_per_month"]) * months


sites = next(
    (
        count
        for count in range(2, len(network) + 1)
        if sum(int(site["candidates"]) for site in network[:count]) >= CANDIDATES
        and sum(screened(site) for site in network[:count]) >= SCREENED
    ),
    len(network),
)
study = dict(
    phase="early_safety",
    sites=sites,
    arms=[dict(compound=compound, dose="standard", participants=30, name="active")],
    control_participants=30,
    follow_up_days=56,
    assessment_days=[28, 56],
    surveillance_days=[56],
    looks=[dict(deadline_day=365, completed_participants=60)],
)
print("DEMO_STUDY=" + json.dumps(study, sort_keys=True))
