#' Read an authenticated realism observation declaration as data.
#' @param path External JSON declaration path.
#' @param sha256 Expected SHA256 of the exact declaration bytes.
#' @param series_id Declared evaluation series identifier.
#' @return The selected series with its authenticated input binding.
#' @export
read_observation_declaration <- function(path,sha256,series_id) {
  actual <- digest::digest(file=path,algo="sha256")
  if(!identical(actual,sha256)) stop("observation declaration SHA256 mismatch")
  declaration <- jsonlite::fromJSON(path,simplifyVector=FALSE)
  if(identical(declaration$schema_id,"trialagentbench.public_verification_declarations/v1"))
    declaration <- declaration$observation_law
  at <- which(vapply(declaration$series,`[[`,"","evaluation_series_id")==series_id)
  if(length(at)!=1L) stop("observation series must have exactly one declaration")
  result <- declaration$series[[at]]
  attr(result,"declared_input") <- list(path=path,sha256=actual,series_index=at-1L,
    schema=declaration$schema,source_citations=declaration$source_citations)
  class(result) <- c("tab_observation_declaration","list")
  result
}

observation_field <- function(declaration,field) {
  paste0("series[",attr(declaration,"declared_input")$series_index,"].",field)
}

observation_unavailable <- function(parameter,arm,declared,field,message) {
  list(parameter=parameter,arm=arm,declared=declared,declaration_field=field,
    estimate=NULL,se=NULL,z=NULL,p=NULL,availability="unavailable",message=message)
}

released_observation_data <- function(data,declaration) {
  participants <- data$participant_records; events <- data$event_history
  observations <- data$endpoint_observations
  if(!all(c("participant_id","assigned_arm","completion_time") %in% names(participants)) ||
     anyNA(participants[c("participant_id","assigned_arm")]) || anyDuplicated(participants$participant_id))
    stop("invalid raw observation participants")
  endpoint <- declaration$design$endpoint_id
  if(!all(c("participant_id",endpoint) %in% names(observations))) stop("missing raw observation endpoint")
  at <- match(observations$participant_id,participants$participant_id)
  if(anyNA(at)) stop("raw observation participant missing")
  if(!setequal(observations$participant_id,participants$participant_id)) stop("raw observations must cover all participants")
  if(declaration$mechanism=="history_dependent_monotone_endpoint_dropout") {
    if(!all(c("occurrence_time","observed") %in% names(observations))) stop("missing raw contact history")
    return(data.frame(id=observations$participant_id,arm=participants$assigned_arm[at],
      time=observations$occurrence_time,observed=as.integer(as.logical(observations$observed)),y=observations[[endpoint]]))
  }
  if(!all(c("participant_id","event_kind","event_at_contact","occurrence_time") %in% names(events)))
    stop("missing raw withdrawal evidence")
  if(anyNA(events[c("participant_id","event_kind","event_at_contact")]) ||
     any(!events$participant_id %in% participants$participant_id)) stop("invalid raw event evidence")
  # Source trace: _withdrawal_source selects event_kind study_withdrawal and
  # event_at_contact, then the earliest occurrence per participant. The event ID
  # and kind come from realism_withdrawal_events; no reason code is required.
  kinds <- vapply(declaration$realism_withdrawal_events,`[[`,"","kind")
  incident <- events[events$event_kind %in% kinds & as.logical(events$event_at_contact),,drop=FALSE]
  if(any(!is.finite(incident$occurrence_time))) stop("withdrawal occurrence time is not bound")
  first <- tapply(incident$occurrence_time,incident$participant_id,min)
  index <- match(as.character(participants$participant_id),names(first))
  withdrew <- !is.na(index)
  followup <- pmin(participants$completion_time,declaration$design$assessment_horizon_days)
  followup[withdrew] <- pmin(first[index[withdrew]],declaration$design$assessment_horizon_days,
    ifelse(is.na(followup[withdrew]),Inf,followup[withdrew]))
  p <- data.frame(id=participants$participant_id,arm=participants$assigned_arm,followup_days=followup,
    withdrawal=ifelse(withdrew,1L,ifelse(is.na(followup),NA_integer_,0L)))
  # The continuous release exports the terminal endpoint only (declaration
  # limitations), not the source schedule's unexported earlier acquisitions.
  horizon <- declaration$design$assessment_horizon_days
  contacts <- data.frame(id=observations$participant_id,time=horizon,observation_time=horizon,
    observed=as.integer(!is.na(observations[[endpoint]])))
  list(participants=p,contacts=contacts)
}

withdrawal_observation <- function(data,declaration) {
  participants <- data$participants; contacts <- data$contacts
  required <- c("id","arm","followup_days","withdrawal")
  if(!all(required %in% names(participants)) || anyNA(participants[c("id","arm")]) ||
     anyDuplicated(participants$id)) stop("unique participant follow-up records required")
  if(!all(c("id","time","observation_time","observed") %in% names(contacts)) ||
     anyNA(contacts) || !all(contacts$observed %in% 0:1) ||
     any(!is.finite(contacts$time) | !is.finite(contacts$observation_time)) ||
     any(contacts$time<0 | contacts$observation_time<contacts$time) ||
     anyDuplicated(contacts[c("id","time")])) stop("invalid released acquisition contacts")
  at <- match(contacts$id,participants$id)
  if(anyNA(at)) stop("contact participant missing from follow-up")
  hazard <- declaration$derived_parameters$constant_withdrawal_hazard_per_day
  if(!length(hazard) || !all(participants$arm %in% names(hazard))) stop("undeclared withdrawal arm")
  horizon <- declaration$design$assessment_horizon_days
  known_time <- !is.na(participants$followup_days)
  known_event <- !is.na(participants$withdrawal)
  if(any(!is.finite(participants$followup_days[known_time])) ||
     any(participants$followup_days[known_time]<0 | participants$followup_days[known_time]>horizon) ||
     !all(participants$withdrawal[known_event] %in% 0:1)) stop("invalid released withdrawal evidence")
  probability <- plogis(declaration$effective_observation$intercept)
  followup <- participants$followup_days[at]
  # _participating_records retains both occurrence and observation <= earliest
  # study_withdrawal. Acquisition is tested only among those eligible contacts;
  # neither a missing terminal endpoint nor treatment discontinuation is withdrawal.
  eligible <- !is.na(followup) & pmax(contacts$time,contacts$observation_time)<=followup
  components <- list()
  for(arm in names(hazard)) {
    rows <- participants$arm==arm
    acquired <- contacts$observed[eligible & participants$arm[at]==arm]
    a <- binomial_parameter("acquisition",sum(acquired),length(acquired),probability,arm)
    a$estimate <- a$realized
    a$se <- if(a$n>0) sqrt(probability*(1-probability)/a$n) else NULL
    a$z <- if(!is.null(a$se) && a$se>0) (a$estimate-probability)/a$se else NULL
    a$se_method <- "binomial null standard error; central exact binomial p"
    a$declaration_field <- observation_field(declaration,"effective_observation.intercept (inverse logit)")
    a$unknown_followup_contacts <- sum(is.na(followup) & participants$arm[at]==arm)
    components[[length(components)+1L]] <- a
    field <- observation_field(declaration,paste0("derived_parameters.constant_withdrawal_hazard_per_day.",arm))
    if(any(!known_time[rows] | !known_event[rows])) {
      rate <- observation_unavailable("withdrawal_rate",arm,hazard[[arm]],field,
        "Exact released withdrawal indicator and follow-up days are required; missing completion is not withdrawal.")
      rate$unknown_participants <- sum(!known_time[rows] | !known_event[rows])
    } else {
      n <- sum(participants$withdrawal[rows]); exposure <- sum(participants$followup_days[rows])
      if(exposure<=0) stop("withdrawal test needs positive exposure")
      value <- n/exposure; se <- sqrt(hazard[[arm]]/exposure)
      rate <- list(parameter="withdrawal_rate",arm=arm,declared=hazard[[arm]],
        estimate=value,realized=value,se=se,z=(value-hazard[[arm]])/se,
        p=poisson.test(n,T=exposure,r=hazard[[arm]])$p.value,n=sum(rows),numerator=n,exposure_days=exposure,
        se_method="Poisson null standard error; exact Poisson rate p",declaration_field=field,
        availability="available")
    }
    components[[length(components)+1L]] <- rate
  }
  parameter_result(components)
}

history_observation <- function(data,declaration) {
  model <- declaration$effective_observation
  effects <- model$effects
  history <- Filter(function(e) identical(e$input_variable,"prior_observed_outcome"),effects)
  arm <- Filter(function(e) identical(e$grouping_variable,"assigned_arm"),effects)
  if(length(history)!=1L || length(arm)!=1L || history[[1]]$power!=1 ||
     history[[1]]$scale<=0) stop("unsupported declared observation effects")
  history <- history[[1]]; arm <- arm[[1]]
  required <- c("id","time","arm","observed","y")
  if(!all(required %in% names(data)) || anyNA(data[c("id","time","arm","observed")]) ||
     !all(data$observed %in% 0:1) || !all(data$arm %in% unlist(arm$levels)) ||
     any(!is.finite(data$time)) || anyDuplicated(data[c("id","time")])) stop("invalid released history contacts")
  data <- data[order(data$id,data$time),,drop=FALSE]
  parts <- split(seq_len(nrow(data)),data$id)
  first_bad <- return_bad <- character()
  keep <- rep(FALSE,nrow(data)); prior <- rep(NA_real_,nrow(data))
  for(rows in parts) {
    if(length(unique(data$arm[rows]))!=1L) stop("participant changes assigned arm")
    # All contacts, including missing ones, must be supplied. S13 uses realized
    # times within the declared windows, never an invented regular time grid.
    nominal <- unlist(declaration$design$contact_times_days)
    early <- declaration$contact_source$early_windows
    late <- declaration$contact_source$late_windows
    if(is.null(early)) early <- rep(0,length(nominal))
    if(is.null(late)) late <- rep(0,length(nominal))
    if(length(rows)!=length(nominal) || any(data$time[rows]<nominal-unlist(early) |
       data$time[rows]>nominal+unlist(late))) stop("incomplete or off-schedule released contact history")
    observed <- data$observed[rows]
    if(isTRUE(model$observe_first_contact) && observed[1]!=1) first_bad <- c(first_bad,as.character(data$id[rows[1]]))
    if(isTRUE(model$monotone_dropout) && any(diff(observed)>0)) return_bad <- c(return_bad,as.character(data$id[rows[1]]))
    if(length(rows)>1L) {
      candidates <- rows[-1L]
      at_risk <- cumprod(observed)[-length(rows)]==1
      keep[candidates] <- at_risk
      prior[candidates] <- data$y[rows[-length(rows)]]
      if("availability_time" %in% names(data) && any(at_risk &
        (is.na(data$availability_time[rows[-length(rows)]]) |
         data$availability_time[rows[-length(rows)]]>data$time[candidates])))
        stop("previous observed outcome is not available at the next contact")
    }
  }
  deterministic <- list(
    list(parameter="observe_first_contact",declaration_field=observation_field(declaration,"effective_observation.observe_first_contact"),
      pass=length(first_bad)==0L,failures=first_bad),
    list(parameter="monotone_dropout",declaration_field=observation_field(declaration,"effective_observation.monotone_dropout"),
      pass=length(return_bad)==0L,failures=return_bad))
  d <- data[keep,,drop=FALSE]
  d$prior_standardized <- (prior[keep]-history$center)/history$scale
  if(!nrow(d) || any(!is.finite(d$prior_standardized))) stop("missing previous observed outcome on declared scale")
  levels <- unlist(arm$levels)
  d$arm <- factor(d$arm,levels=levels)
  declared <- setNames(model$intercept+unlist(arm$coefficients),paste0("arm",levels))
  declared <- c(declared,prior_standardized=history$coefficient)
  fit <- glm(observed ~ 0+arm+prior_standardized,data=d,family=binomial(),
    control=glm.control(epsilon=1e-10,maxit=100))
  components <- wald_components(fit,declared)
  for(i in seq_along(components)) {
    components[[i]]$arm <- if(i<=length(levels)) levels[i] else NULL
    components[[i]]$declaration_field <- observation_field(declaration,if(i<=length(levels))
      paste0("effective_observation.intercept + effective_observation.effects[effect_id=",arm$effect_id,"].coefficients[",levels[i],"]") else
      paste0("effective_observation.effects[effect_id=",history$effect_id,"].coefficient"))
    components[[i]]$n <- nrow(d)
  }
  result <- parameter_result(components)
  result$deterministic <- deterministic
  result$standardization <- list(center=history$center,scale=history$scale,
    declaration_field=observation_field(declaration,paste0("effective_observation.effects[effect_id=",history$effect_id,"].{center,scale}")))
  result$not_operative <- list(list(parameter="process_profile observation effects (including bmi)",
    reason="The entire earlier observation law is replaced by effective_observation.",
    declaration_field=observation_field(declaration,"composition_order")))
  result
}

#' Test the declared composite realism observation process.
#' @param data S10/S11 list of participants and contacts, or S12/S13 contact rows.
#' @param declaration Authenticated series from read_observation_declaration.
#' @return Unadjusted component tests, deterministic checks and input binding.
#' @export
realism_observation <- function(data,declaration) {
  if(!inherits(declaration,"tab_observation_declaration")) stop("authenticated observation declaration required")
  if(is.list(data) && "participant_records" %in% names(data)) data <- released_observation_data(data,declaration)
  result <- switch(declaration$mechanism,
    continuous_time_withdrawal_plus_intermittent_measurement=withdrawal_observation(data,declaration),
    history_dependent_monotone_endpoint_dropout=history_observation(data,declaration),
    stop("unsupported declared realism observation mechanism"))
  result$declared_input <- attr(declaration,"declared_input")
  result
}
