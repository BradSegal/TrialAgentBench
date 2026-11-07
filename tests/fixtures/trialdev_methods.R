# Executable controls for the full population's measured baseline strata.
library(trialagentbenchR)
method <- commandArgs(trailingOnly = TRUE)[1]
source <- read_trialagentbench_parquet("data/observational_cohort.parquet")
q <- prop.table(table(source$risk))
rows <- list()
for (asset in c("asset_alpha", "asset_beta")) {
  pair <- source[source$asset_id %in% c(asset, "control"), ]
  treated <- as.numeric(pair$asset_id == asset)
  x <- pair$risk
  for (endpoint in c("clinical_outcome", "safety_risk")) {
    column <- if (endpoint == "clinical_outcome") "outcome" else "safety"
    unit <- if (endpoint == "clinical_outcome") "points" else "proportion"
    y <- pair[[column]]
    if (method == "regression_standardisation") {
      design <- cbind(1, x, treated, treated * x)
      fit <- lm.fit(design, y)
      bread <- solve(crossprod(design))
      covariance <- nrow(design) / (nrow(design) - ncol(design)) *
        bread %*% crossprod(design * fit$residuals) %*% bread
      gradient <- c(0, 0, 1, mean(source$risk))
      estimate <- as.numeric(gradient %*% fit$coefficients)
      se <- sqrt(as.numeric(gradient %*% covariance %*% gradient))
    } else {
      estimate <- 0
      variance <- 0
      for (stratum in names(q)) {
        mask <- x == as.numeric(stratum)
        a <- y[mask & treated == 1]
        c0 <- y[mask & treated == 0]
        share <- as.numeric(q[[stratum]])
        if (method == "matching") {
          estimate <- estimate + share * mean(outer(a, c0, "-"))
        } else if (method == "inverse_probability_weighting") {
          propensity <- length(a) / (length(a) + length(c0))
          target_weight <- share / mean(mask)
          estimate <- estimate + sum(target_weight * (
            treated[mask] * y[mask] / propensity -
            (1 - treated[mask]) * y[mask] / (1 - propensity)
          )) / nrow(pair)
        } else stop("Unknown prespecified method")
        variance <- variance + share^2 * (var(a) / length(a) + var(c0) / length(c0))
      }
      se <- sqrt(variance)
    }
    stopifnot(is.finite(estimate), is.finite(se), se > 0)
    margin <- qnorm(0.975) * se
    rows[[length(rows) + 1]] <- list(
      asset_id = asset, endpoint = endpoint, unit = unit,
      estimate = estimate, standard_error = se,
      sampling_interval = list(lower = estimate - margin, upper = estimate + margin,
        confidence_level = 0.95, interval_type = "central_frequentist_confidence_interval")
    )
  }
}
result <- list(
  stream = "portfolio_management", action = "nominate", asset_id = "asset_alpha",
  comparative_analysis = list(analysis_method = method, estimates = rows),
  identification_limit_acknowledged = TRUE, preference_dependence_acknowledged = TRUE,
  uncertainty_statement = "Conditional on declared baseline strata; marginal 95% endpoint intervals only.",
  rationale = "Evaluate the favorable candidate under the public costs and safety ceiling."
)
jsonlite::write_json(result, "scratch/analysis_result.json", auto_unbox = TRUE, digits = 17)
