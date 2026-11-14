# Verification

TrialAgentBench v1.1.0 (Hugging Face revision `5fdbecf941e3a04bec4070abaed44efcab81f2ab`, scorer `644c4d60`) was checked against its own declarations, and its analyses and published scores were reproduced independently in R 4.5.1. Every check passed. [Methods](METHODS.md) defines the task contexts C1–C5, assumption tiers A1–A3 and credited analysis routes.

## The released data match their declarations

| Checked | Result |
| --- | --- |
| Cell and series identities; exclusion of withheld cells from the public split; task and replay counts | 38/38, 1,260/1,260 and 945/945, exact |
| C5 consistency and C2–C5 reconstructions against C1 | 65,390/65,390; numerical reconstructions within 1e-12 × max(1, \|x\|), other checks exact |
| Design evidence, estimand and design bindings, baseline support | 126,232/126,232, 3,780/3,780 and 161,200/161,200, exact |
| Contact and monitoring schedules | 28,078/28,078; familywise α within 0.0001 of 0.05 |
| Diagnostic instructions, observation indicators and intercurrent events | 9,424/9,424, 9,300/9,300 and 1,488/1,488, exact |

## The declared statistical properties hold

On the public split, no declared property was rejected. Each family was tested against its declared margins, with Holm correction at α = 0.05.

| Property family | Tests | Margin |
| --- | --- | --- |
| Measurement mechanisms | 88 in 10 cells | declared parameter values |
| Observation processes | 14 in 4 cells | declared parameter values |
| Intercurrent events | 8 in 2 cells | declared parameter values |
| Censoring | 5 in 5 cells | declared parameter values |
| Calendar drift | 1 in 1 cell | declared parameter values |
| Credited-route recovery | 21 composites over 25 routes | bias 0.2 SD pooled, 0.3 SD by effect; coverage 0.88; fit support 0.8 |
| A3 practical consequences | 4 in 4 cells | shift 1 SD; inner bound 0.8 SD or width ratio 0.7; no inversion; benign twin |
| A2 decision regret | 25 (5 cells × 5 economics points) | mean regret at least 1.25 × US$5M × the declared value multiplier |

The A3 consequence test correctly rejects in all 4 positive controls, where the property is absent.

## Analyses and scores reproduce in R

| Reproduced | Result |
| --- | --- |
| Stored replay references, credited key values and plan readouts | 2,439/2,439 within declared tolerances |
| External-reference routes, at release source `3c113f53` | 18/18 routes; 77/77 comparisons |
| Refits of 33 real trials (17,100 participants) | 165/165; atol 1e-10, rtol 1e-08 |
| Group-sequential monitoring dataset | Matches stored readouts at 6-decimal precision |
| TrialDev negative-binomial fits | 59/59 estimates and standard errors |
| Re-scoring from the retained raw submissions | 5,241/5,241 published values: 5,134 numeric within relative 1e-12 (4,978 bit-exact), 107 displays at printed precision |
| Leaderboard, given the pipeline baseline's 84 stored task scores | 177/177 effects, intervals and ranks: 96 bit-exact, 81 at printed precision |

## Realism

In the A3 cells, all 14 declared measurement parameters and implied quantities with a cited range fall inside it.
