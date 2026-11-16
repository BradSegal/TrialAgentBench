\name{scoring}
\alias{scoring}
\alias{score_compare_exact}
\alias{score_compare_output}
\alias{score_display_tokens}
\alias{score_raw_ledger}
\alias{score_raw_bank}
\alias{score_oracle_denominators}
\alias{score_submission_failure}
\alias{score_load_declaration}
\alias{score_pcg64_hex}
\alias{score_pcg64_integers}
\alias{score_pcg64_raw}
\alias{score_pcg64}
\alias{score_rule}
\alias{score_seed_sequence}
\alias{score_share_interval}
\alias{score_trialdev_approval}
\alias{score_trialdev_study_cost}
\alias{score_trialdev_value}
\alias{score_trialeval_complete}
\alias{score_trialeval_error}
\alias{score_trialeval_impute}
\alias{score_trialeval_model_effects}
\alias{score_trialeval_serious_miss}
\alias{score_world_bootstrap}
\alias{score_world_strata}
\alias{score_write_csv}
\alias{score_write_json}
\title{Independent declaration-bound TrialAgentBench re-scoring}
\description{
Independent implementations of the authenticated TrialAgentBench scoring rules.
The declaration supplies benchmark constants; supplied reports, outcomes and
accounting facts supply observations. The runner in \file{exec/verify.R} checks
the declaration digest before computing a score.
}
\details{
The PCG64 generator implements NumPy SeedSequence integer expansion, a 128-bit
linear congruential state and the XSL RR 64-bit output permutation using exact
16-bit limbs. Bounded draws use the buffered 32-bit Lemire algorithm.
The statistical functions accept an authenticated declaration as their final
argument. Computation and comparison are separate executable phases.
The published leaderboard verification is conditional on the test-then-correct analysis's 84 stored task scores.
}
\keyword{internal}
