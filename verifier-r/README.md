# Independent R verification

`tabverify` implements the TrialAgentBench analysis and scoring rules independently in R. It fits declared analyses, checks observation laws and recomputes scores from supplied observations. The verification results and their scope are in [Verification](../docs/VERIFICATION.md).

Install with R 4.5.1 and the dependencies listed in `DESCRIPTION`:

```sh
R CMD INSTALL verifier-r
```

`package-versions.json` records the environment used for the verification results. Its image digest identifies that checked environment; the source package installs directly into an R environment.

The single runner accepts an analysis request directory or a scoring request:

```sh
Rscript verifier-r/exec/verify.R request "$REQUEST" "$OUTPUT"
Rscript verifier-r/exec/verify.R rescore verification/scoring-rules.json \
  "$DECLARATION_SHA256" "$REQUEST_JSON" "$OUTPUT_JSON"
```

The declaration digest must come from the release checksum file. The runner refuses an altered declaration or an existing scoring output. Request tables use relative CSV paths. Responses preserve fit failures and include the R and dependency versions.

Scoring operations cover TrialEval error, missing-answer completion, model effects and world bootstrap, and TrialDev study costs, approval, value and oracle-share intervals. Benchmark constants are read from the authenticated scoring declaration. The published leaderboard verification is conditional on the test-then-correct analysis's 84 stored task scores; it is a verification result, rather than an end-to-end rerun from the public dataset alone.

Observation-process checks use released contact histories and the matching declared observation law. For S10–S13 a missing declaration is reported as unavailable. The caller owns participant/world namespaces and Holm adjustment.
