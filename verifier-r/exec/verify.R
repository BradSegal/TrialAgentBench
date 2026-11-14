#!/usr/bin/env Rscript
# Run a declared analysis or independently score supplied task observations.
args <- commandArgs(TRUE)
if (!length(args)) stop("Use: verify.R request INPUT OUTPUT, or verify.R rescore DECLARATION SHA256 INPUT OUTPUT")
if (identical(args[1], "request")) {
  if (length(args) != 3L) stop("request requires an input directory and a new output directory")
  tabverify::run(args[2], args[3])
} else if (identical(args[1], "rescore")) {
  if (length(args) != 5L) stop("rescore requires a declaration, SHA256, request JSON and new output JSON")
  if (file.exists(args[5])) stop("output already exists")
  declaration <- tabverify::score_load_declaration(args[2], args[3])
  request <- jsonlite::fromJSON(args[4], simplifyVector = FALSE)
  if (!is.character(request$operation) || length(request$operation) != 1L) stop("one scoring operation required")
  number <- function(name, nullable = FALSE) {
    value <- request[[name]]
    if (is.null(value) && nullable) return(NA_real_)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) stop("finite scalar required: ", name)
    value
  }
  vector <- function(value) vapply(value, function(item) {
    if (is.null(item)) return(NA_real_)
    if (!is.numeric(item) || length(item) != 1L || !is.finite(item)) stop("numeric vector required")
    as.double(item)
  }, 0.0)
  matrix_values <- function() {
    if (!length(request$values) || length(unique(lengths(request$values))) != 1L) stop("rectangular values required")
    value <- do.call(rbind, lapply(request$values, vector))
    if (!is.matrix(value) || !length(value)) stop("nonempty row-major values matrix required")
    if (!is.null(request$model_ids) || !is.null(request$task_ids)) {
      dimnames(value) <- list(unlist(request$model_ids), unlist(request$task_ids))
    }
    value
  }
  result <- switch(request$operation,
    trialeval_error = {
      confidence <- number("confidence")
      if (confidence <= 0 || confidence >= 1) stop("confidence must be between zero and one")
      tabverify::score_trialeval_error(request$reports, vector(request$truth), number("reference_se"),
        request$measure, declaration, confidence, isTRUE(request$bank_scale))
    },
    trialeval_imputation = tabverify::score_trialeval_impute(vector(request$errors), number("truth"),
      number("reference_se"), request$measure),
    trialeval_completion = tabverify::score_trialeval_complete(number("error", TRUE), number("share"), number("penalty")),
    trialeval_serious_miss = tabverify::score_trialeval_serious_miss(number("error"), number("reference"), declaration),
    trialeval_model_effects = tabverify::score_trialeval_model_effects(matrix_values(), declaration),
    trialeval_world_effects = tabverify::score_world_bootstrap(matrix_values(), request$keys, declaration),
    trialdev_cost = tabverify::score_trialdev_study_cost(request$commitments, request$enrollments, declaration),
    trialdev_approval = tabverify::score_trialdev_approval(lapply(request$active, vector),
      lapply(request$control, vector), number("treated_total"), declaration),
    trialdev_value = tabverify::score_trialdev_value(isTRUE(request$approved), number("outcome", !isTRUE(request$approved)),
      number("safety", !isTRUE(request$approved)), number("study_cost"), number("day"), request$preferences, declaration),
    trialdev_share_interval = tabverify::score_share_interval(vector(request$value), vector(request$oracle),
      unlist(request$strata, use.names = FALSE), declaration),
    stop("unknown re-scoring operation")
  )
  tabverify::score_write_json(result, args[5])
} else stop("mode must be request or rescore")
