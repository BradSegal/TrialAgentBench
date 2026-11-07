library(trialagentbenchR)
read_case <- function(cell) read_trialagentbench_parquet(paste0('data/', cell, '.parquet'))

clustered <- read_case('clustered')
survival_fit <- survival::survfit(
    survival::Surv(observed_time, event_observed) ~ assigned_arm,
    data=clustered, cluster=cluster_id, robust=TRUE
)
survival_summary <- summary(survival_fit, times=180)
stopifnot(length(survival_summary$surv) == 2L,
          all(is.finite(survival_summary$surv)),
          all(is.finite(survival_summary$std.err)),
          all(survival_summary$std.err > 0))

longitudinal <- read_case('longitudinal')
longitudinal$assigned_arm <- relevel(factor(longitudinal$assigned_arm), ref='control')
longitudinal$visit <- factor(longitudinal$occurrence_time)
mixed_fit <- nlme::lme(
    asthma_control_score ~ assigned_arm * visit + age_years + baseline_body_mass_index,
    random=~1 | participant_id, data=longitudinal, na.action=na.omit
)
stopifnot(all(is.finite(nlme::fixef(mixed_fit))),
          all(is.finite(vcov(mixed_fit))), all(diag(vcov(mixed_fit)) > 0))

counts <- read_case('counts')
counts$assigned_arm <- relevel(factor(counts$assigned_arm), ref='control')
count_fit <- MASS::glm.nb(
    heart_failure_hospitalization_count ~ assigned_arm + age_years + baseline_body_mass_index, data=counts
)
stopifnot(count_fit$converged, count_fit$theta > 0,
          all(is.finite(coef(count_fit))), all(is.finite(vcov(count_fit))))

write.csv(data.frame(
    model=c('clustered_survival', 'longitudinal_random_intercept', 'negative_binomial'),
    participants=c(nrow(clustered), length(unique(longitudinal$participant_id)), nrow(counts)),
    observations=c(nrow(clustered), nobs(mixed_fit), nobs(count_fit)),
    package=c('survival', 'nlme', 'MASS'),
    version=vapply(c('survival', 'nlme', 'MASS'), function(x) as.character(packageVersion(x)), '')
), 'scratch/design_models.csv', row.names=FALSE)
