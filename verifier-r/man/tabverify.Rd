\name{tabverify}
\alias{tabverify}
\alias{estimate}
\alias{route_configuration}
\alias{assign_delete_groups}
\alias{derive_covariates}
\alias{ice_contacts}
\alias{compare_reference}
\alias{mechanism}
\alias{mechanism_parameters}
\alias{observation_coefficients}
\alias{read_observation_declaration}
\alias{realism_observation}
\alias{ice_coefficients}
\alias{loss_rate}
\alias{calendar_coefficient}
\alias{pool_truth}
\alias{read_payload}
\alias{write_result}
\alias{run}
\title{Independent TrialAgentBench analyses and property checks}
\description{Fits declared analyses from released CSV tables and performs pooled parameter falsification tests in a pinned, offline R environment. No construction, generator or harness source is loaded.}
\usage{
route_configuration(cell_id, route_id,
  routes_file = system.file("routes.csv", package = "tabverify"))
assign_delete_groups(data, cfg)
derive_covariates(data, profile)
ice_contacts(participants, events, contact_times)
estimate(data, cfg, tables = list())
compare_reference(result, stored, kind, ratio = FALSE, key = FALSE)
mechanism(data, endpoint_type = c("binary", "count"))
mechanism_parameters(data, strata, endpoint_type = c("binary", "count"),
  recorded_domain = FALSE)
observation_coefficients(data, declaration, profile = NULL, series_id = NULL,
                         composite_process = NULL)
read_observation_declaration(path, sha256, series_id)
realism_observation(data, declaration)
ice_coefficients(data, declaration, profile = NULL)
loss_rate(data, hazard, process_multiplier = 1, cell_multiplier = 1,
          cluster = FALSE)
calendar_coefficient(data, offset, accrual_days, profile = NULL)
pool_truth(...)
read_payload(dir)
write_result(path, result)
run(dir, out)
}
\arguments{
\item{cell_id, route_id}{Exported cell and route identifiers.}
\item{routes_file}{Route CSV containing operative defaults and optional validated convention overrides.}
\item{profile}{Exported clinical profile with baseline mappings and prognostic population constants.}
\item{participants, events}{Released raw participant records and event history.}
\item{contact_times}{Declared contact schedule in days.}
\item{series_id}{Declared series identifier. S10--S13 require an authenticated realism declaration.}
\item{composite_process}{Optional function(data, declaration) returning composite realism tests for S10--S13; otherwise preparing contact evidence.}
\item{sha256}{Expected SHA256 of the external observation declaration bytes. Mismatch refuses loading.}
\item{data}{Released analysis table, or canonical contact/validation rows.}
\item{cfg}{Resolved procedure or explicit R-panel configuration.}
\item{tables}{Named auxiliary tables. Sequential routes require the monitoring plan.}
\item{result}{Named estimate vector for comparison, or a protocol result for serialization.}
\item{stored}{Named stored estimate and bounds. Replay ratios are on the log scale; keys are in native units.}
\item{kind}{R-panel analysis kind.}
\item{ratio}{Whether the contrast is a ratio.}
\item{key}{Whether stored values come from credited or plan_replay key rows. Only sequential key values are rounded to six decimals.}
\item{endpoint_type}{Binary misclassification or count capture.}
\item{strata}{Exported validation strata, decoded as a list of lists.}
\item{recorded_domain}{Test selection and accuracy on eligible recorded endpoints, retaining excluded counts. Selection is independent of recording within stratum: patient_worlds/endpoint_validation.py:411--416 draws from its own seeded stream before withdrawal filtering at :234--272.}
\item{declaration}{Exported observation or intercurrent-event declaration.}
\item{hazard}{Declared annual loss hazard, or NULL if undeclared.}
\item{process_multiplier, cell_multiplier}{Declared hazard and time-scale multipliers.}
\item{cluster}{Whether cluster frailty precludes a declared marginal loss rate.}
\item{offset}{Total calendar log-hazard change.}
\item{accrual_days}{Declared accrual window.}
\item{...}{Reserved for the later TrialDev pool-truth specification.}
\item{dir}{Request directory containing request.json and relative CSV files.}
\item{path}{Response JSON path, or external observation declaration path.}
\item{out}{Output directory for response.json and sessionInfo.txt.}
}
\details{Selection shares pool counts within stratum across arms. Sensitivity, specificity and capture pool counts within arm and stratum. Their tests use the central exact binomial p-value, twice the smaller inclusive tail capped at one. Observation and discontinuation models use contact-level logistic maximum likelihood and Wald z tests. ICE contacts are strictly after risk_window_start and through the earlier of risk_window_end and observed_time, including the discontinuation contact. Loss testing uses pooled person-time through loss, endpoint or administrative end; exposure_days are divided by 365.25. Calendar testing fits a Breslow Cox model stratified by world with a separate intervention term per task, bmi_curve, entry divided by the accrual window and robust covariance clustered on world times cluster. No family multiplicity correction occurs here. The caller owns Holm adjustment and deterministic observation/exposure checks.

Jackknife analysis sets are restricted before raw SHA256 ordering of decimal unit identifiers within arms. Labels cycle through the declared group count. S08 instead deletes one whole cluster at a time. Every nuisance model is refitted and censoring support is checked through the horizon even when no endpoint occurs. S07 uses participant-size-weighted within-cluster IPCW curves and within-arm censoring Cox fits. Intervals and covariates resolve from operative route CSV defaults or validated override columns. Covariate overrides must identify released baseline columns; formula text is rejected. Interval overrides allow normal or fitted-model residual t intervals, with no numeric degrees of freedom. Exported-text disagreements are documented in convention_note. Joint likelihood requires at least 25 audited participants per arm-stratum and audited events/non-events per stratum; complex-step observed information must be positive definite and is inverted in full. Zero audited false negatives or false positives set a non-refusing accuracy nuisance at probability boundary diagnostic, also exposed in the request response message. S05 binds its predictors to released dictionary roles and checks reference_population_member against its protocol BMI threshold; the input hashes are retained in route_inputs.json.

Explicit panel configurations also support MMRM and NB2; these are ports for the separate numerical panel, not additional declared TrialEval routes. The pool_truth function raises an explicit pending-specification error.}
\value{An estimate vector, comparison data frame, configuration, group labels, decoded request or discriminated protocol result. run and write_result return their result invisibly. Statistical checks return component tests with counts, estimates and availability.}
\section{Realism observation}{The authenticated series supplies every parameter. Raw input can be a list named participant_records, event_history and endpoint_observations. Withdrawal is the earliest incident event of the declared kind; it needs no reason code. Follow-up uses that exact occurrence or the released completion_time, capped at the declared horizon. The endpoint field is selected from design.endpoint_id.

Canonical S10/S11 data is a list: participants has unique id, arm, followup_days and withdrawal (0/1, unknown as NA); contacts has id, time, observation_time and observed (0/1). Follow-up ends at the earliest study withdrawal or administrative horizon. Treatment discontinuation and missing endpoints do not define withdrawal. Acquisition counts only contacts whose occurrence and observation times are through known follow-up. Continuous release supplies terminal contacts only. Missing exact follow-up or withdrawal indicators make the arm's rate unavailable. Acquisition uses central exact binomial tails; withdrawal uses an exact exposure-based Poisson test. Their SE and z use null variances, labelled in each result.

S12/S13 data has id (unique across pooled worlds), arm, realized time, observed and y on the declared endpoint scale, with every scheduled contact retained. First observation and absorbing loss are checked explicitly. Logistic fitting includes only post-first contacts observed through the preceding contact, with arm intercepts and the standardized previous observed y. An optional availability_time verifies the previous outcome was available. Wald tests use fitted coefficient covariance. The old observation effects, including S13 BMI, are reported as not operative. Component declaration fields and authenticated input provenance are retained. No Holm or other multiplicity adjustment is applied.}
