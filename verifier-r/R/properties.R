parameter_result <- function(components, message="") {
  list(type="mechanism",p_diff=NULL,components=components,
       availability=if(length(components)) "available" else "unavailable",message=message)
}

binomial_parameter <- function(parameter, numerator, denominator, declared, arm=NULL, stratum=NULL) {
  if(!is.finite(declared) || declared<0 || declared>1) stop("invalid declared probability")
  if(numerator<0 || numerator>denominator || numerator!=round(numerator) || denominator!=round(denominator))
    stop("invalid binomial counts")
  list(parameter=parameter,arm=arm,stratum=stratum,numerator=numerator,n=denominator,
       realized=if(denominator>0) numerator/denominator else NULL,declared=declared,
       p=if(denominator>0) min(1,2*min(pbinom(numerator,denominator,declared),
         pbinom(numerator-1,denominator,declared,lower.tail=FALSE))) else NULL,
       availability=if(denominator>0) "available" else "unavailable")
}

#' Test pooled validation-process parameters against declarations.
#' @param data Canonical rows: treatment (0/1), stratum, selected, y, truth.
#' @param strata Exported endpoint_validation_strata, decoded without simplification.
#' @param endpoint_type Binary misclassification or count capture.
#' @param recorded_domain Test selection and accuracy on eligible recorded endpoints, retaining excluded counts.
#' @return Mechanism result with exact binomial tests and pooled counts.
#' @export
mechanism_parameters <- function(data, strata, endpoint_type=c("binary","count"),recorded_domain=FALSE) {
  endpoint_type <- match.arg(endpoint_type)
  required <- c("treatment","stratum","selected","y","truth")
  if(!all(required %in% names(data))) stop("missing mechanism columns")
  if(recorded_domain) {
 # Read-only source trace: patient_worlds/endpoint_validation.py:411-416
 # draws selection from its own seeded stream on the full population, before
 # :234-272 limits recordings by withdrawal. Within stratum, selection is
 # independent of recording, so its share is identifiable on recorded rows.
 eligible <- !is.na(data$y) & !is.na(data$stratum) & !is.na(data$selected)
 excluded <- data[!eligible,,drop=FALSE]
 counts <- as.data.frame(table(arm=ifelse(excluded$treatment==1,"active","control"),
   stratum=ifelse(is.na(excluded$stratum),"unreleased",excluded$stratum)),stringsAsFactors=FALSE)
 counts <- counts[counts$Freq>0,,drop=FALSE]
 result <- mechanism_parameters(data[eligible,,drop=FALSE],strata,
   endpoint_type)
 note <- paste0("MECH eligible recorded endpoint rows ",sum(eligible),"/",nrow(data),
   "; excluded cell rows ",sum(!eligible),"; excluded arm/stratum counts ",
   jsonlite::toJSON(counts,dataframe="rows",auto_unbox=TRUE),
   "; selection is independent of recording within stratum: own seeded stream before withdrawal; source patient_worlds/endpoint_validation.py:411-416,234-272; release_projection/tables.py:847-859.")
 for(i in seq_along(result$components)) {
  component <- result$components[[i]]
  component$message <- note
  result$components[[i]] <- component
 }
 result$domain_counts <- list(total=nrow(data),eligible=sum(eligible),excluded=sum(!eligible),by_arm_stratum=counts)
   return(result)
  }
  if(anyNA(data[c("treatment","stratum","selected")])) stop("missing mechanism labels")
  if(!all(data$treatment %in% 0:1)) stop("invalid treatment labels")
  data$selected <- as.logical(data$selected)
  if(anyNA(data$selected)) stop("invalid selection flag")
  declared_strata <- vapply(strata, `[[`, "", "value")
  if(length(setdiff(unique(data$stratum), declared_strata))) stop("undeclared validation stratum")
  v <- data$selected & !is.na(data$y) & !is.na(data$truth)
  components <- list()
  for(s in strata) {
    rows <- data$stratum==s$value
    components[[length(components)+1]] <- binomial_parameter("selection",sum(data$selected[rows]),sum(rows),s$validation_probability,stratum=s$value)
    for(arm in c("control","active")) {
      cell <- data[rows & v & data$treatment==as.integer(arm=="active"),,drop=FALSE]
      if(endpoint_type=="binary") {
        if(!all(c(cell$y,cell$truth) %in% 0:1)) stop("invalid binary evidence")
        a <- if(length(s$accuracy_by_arm)) Filter(function(a) a$arm_id==arm,s$accuracy_by_arm)[[1]] else s
        positive <- cell$truth==1; negative <- cell$truth==0
        components[[length(components)+1]] <- binomial_parameter("sensitivity",sum(cell$y[positive]),sum(positive),a$sensitivity,arm,s$value)
        components[[length(components)+1]] <- binomial_parameter("specificity",sum(1-cell$y[negative]),sum(negative),a$specificity,arm,s$value)
      } else {
        if(any(cell$y<0 | cell$truth<cell$y | cell$y!=round(cell$y) | cell$truth!=round(cell$truth))) stop("invalid capture counts")
        a <- if(length(s$capture_by_arm)) Filter(function(a) a$arm_id==arm,s$capture_by_arm)[[1]] else s
        components[[length(components)+1]] <- binomial_parameter("capture",sum(cell$y),sum(cell$truth),a$capture_probability,arm,s$value)
      }
    }
  }
  parameter_result(components)
}

#' Compare arm-specific measurement mechanisms (descriptive legacy check).
#' @param data Canonical validation rows, pooled over replications.
#' @param endpoint_type Binary endpoint or count endpoint.
#' @return Mechanism result with component p-values and p_diff.
#' @export
mechanism <- function(data, endpoint_type=c("binary","count")) {
  endpoint_type <- match.arg(endpoint_type)
  v <- data[as.logical(data$selected) & !is.na(data$truth) & !is.na(data$y),,drop=FALSE]
  components <- list()
  for(component in if(endpoint_type=="binary") c("sensitivity","specificity") else "capture") {
    rows <- if(component=="sensitivity") v[v$truth==1,,drop=FALSE] else
      if(component=="specificity") v[v$truth==0,,drop=FALSE] else v[v$truth>0,,drop=FALSE]
    if(!nrow(rows) || length(unique(rows$treatment))<2) next
    rows$success <- if(component=="specificity") 1-rows$y else rows$y
    rows$total <- if(component=="capture") rows$truth else 1
    totals <- aggregate(cbind(success,total) ~ treatment+stratum,rows,sum)
    separated <- any(totals$success==0 | totals$success==totals$total)
    if(separated) {
      tab <- array(0,c(2,2,length(unique(rows$stratum))))
      strata <- sort(unique(rows$stratum))
      for(i in seq_len(nrow(totals))) {
        at <- match(totals$stratum[i],strata); arm <- totals$treatment[i]+1L
        tab[arm,,at] <- c(totals$success[i],totals$total[i]-totals$success[i])
      }
      p <- mantelhaen.test(tab,exact=TRUE)$p.value
    } else {
      form <- if(length(unique(rows$stratum))>1) cbind(success,total-success) ~ treatment+factor(stratum) else cbind(success,total-success) ~ treatment
      fit <- glm(form,data=rows,family=binomial())
      reduced <- update(fit,. ~ .-treatment)
      if(!fit$converged || !reduced$converged) stop("mechanism logistic did not converge")
      p <- pchisq(deviance(reduced)-deviance(fit),df=1,lower.tail=FALSE)
    }
    components[[length(components)+1]] <- list(parameter=component,p=p)
  }
  result <- parameter_result(components)
  if(length(components)) result$p_diff <- min(1,length(components)*min(vapply(components,`[[`,numeric(1),"p")))
  result
}

wald_components <- function(fit, declared) {
  if(!isTRUE(fit$converged) && inherits(fit,"glm")) stop("coefficient model did not converge")
  beta <- coef(fit); se <- sqrt(diag(vcov(fit)))
  lapply(names(declared),function(name) {
    valid <- name %in% names(beta) && is.finite(beta[name]) && is.finite(se[name]) && se[name]>0
    z <- if(valid) unname((beta[name]-declared[[name]])/se[name]) else NULL
    list(parameter=name,declared=declared[[name]],estimate=if(valid) unname(beta[name]) else NULL,
         se=if(valid) unname(se[name]) else NULL,z=z,p=if(valid) 2*pnorm(-abs(z)) else NULL,
         availability=if(valid) "available" else "unavailable")
  })
}

contact_risk_set <- function(data, response, monotone=FALSE, first_forced=FALSE) {
  if(!all(c("id","time",response) %in% names(data))) stop("missing contact rows")
  data <- data[order(data$id,data$time),,drop=FALSE]
  # IDs must be unique across pooled worlds; the caller supplies that namespace.
  parts <- split(seq_len(nrow(data)),data$id)
  keep <- rep(TRUE,nrow(data))
  for(rows in parts) {
    if(anyDuplicated(data$time[rows])) stop("duplicate participant contact time")
    if(first_forced) keep[rows[1]] <- FALSE
    if(monotone) {
      first <- which(data[[response]][rows]==0)[1]
      if(!is.na(first) && first<length(rows)) keep[rows[seq.int(first+1,length(rows))]] <- FALSE
    }
  }
  data[keep,,drop=FALSE]
}

logistic_declaration <- function(data, declaration, response, ice=FALSE) {
  declared <- c("(Intercept)"=declaration$intercept,time=declaration$time_coefficient)
  columns <- "time"
  if(ice || (!is.null(declaration$baseline_coefficient) && declaration$baseline_coefficient!=0)) {
    baseline <- if(is.null(declaration$baseline_variable)) "bmi_curve" else declaration$baseline_variable
    columns <- c(columns,baseline); declared[baseline] <- declaration$baseline_coefficient
  }
  if(ice) {
    arm <- setNames(vapply(declaration$arm_coefficients,`[[`,numeric(1),"value"),vapply(declaration$arm_coefficients,`[[`,"","arm_id"))
    columns <- c(columns,"treatment"); declared["treatment"] <- arm[["active"]]-arm[["control"]]
    declared["(Intercept)"] <- declared["(Intercept)"]+arm[["control"]]
  }
  for(effect in declaration$effects) {
    if(effect$effect_type!="main") stop("unsupported declared logistic effect type")
    columns <- c(columns,effect$input_variable)
    declared[effect$input_variable] <- effect$coefficient
  }
  if(!all(c(columns,response) %in% names(data))) stop("missing declared logistic covariate")
  if(anyNA(data[c(columns,response)])) stop("missing logistic evidence")
  if(!all(data[[response]] %in% 0:1)) stop("invalid logistic response")
  fit <- glm(reformulate(unique(columns),response),data=data,family=binomial(),control=glm.control(epsilon=1e-10,maxit=100))
  parameter_result(wald_components(fit,declared))
}

#' Fit declared contact-level observation coefficients.
#' @param data Contact rows with id, time, observed and declared covariates.
#' @param declaration Exported observation model.
#' @param profile Optional exported clinical profile for covariate derivation.
#' @param series_id Declared series identifier.
#' @param composite_process Optional function(data, declaration) testing a bound realism law.
#' @return Wald z tests for each declared coefficient.
#' @export
observation_coefficients <- function(data,declaration,profile=NULL,series_id=NULL,composite_process=NULL) {
  if(is.null(series_id) && inherits(declaration,"tab_observation_declaration")) series_id <- declaration$evaluation_series_id
  if(!is.null(series_id) && series_id %in% paste0("TE-S",10:13)) {
    if(!inherits(declaration,"tab_observation_declaration")) return(parameter_result(list(),"declaration incomplete"))
    if(!identical(series_id,declaration$evaluation_series_id)) stop("observation series binding mismatch")
    if(is.null(composite_process)) return(realism_observation(data,declaration))
    return(composite_process(data,declaration))
  }
  if(!is.null(composite_process)) data <- composite_process(data,declaration)
  if(!is.null(profile)) data <- derive_covariates(data,profile)
  data <- contact_risk_set(data,"observed",isTRUE(declaration$monotone_dropout),isTRUE(declaration$observe_first_contact))
  if(isTRUE(declaration$terminal_missingness_only)) {
    if(!"terminal" %in% names(data)) stop("terminal contact indicator required")
    data <- data[as.logical(data$terminal),,drop=FALSE]
  }
  logistic_declaration(data,declaration,"observed")
}

#' Fit declared at-risk discontinuation coefficients.
#' @param data Contact rows with id, time, event, treatment and bmi_curve.
#' @param declaration Exported intercurrent-event declaration.
#' @param profile Optional exported clinical profile for covariate derivation.
#' @return Wald z tests for each declared coefficient.
#' @export
ice_coefficients <- function(data,declaration,profile=NULL) {
  if(!is.null(profile)) data <- derive_covariates(data,profile)
  if(!"observed_time" %in% names(data) || anyNA(data$observed_time)) stop("observed_time is required for ICE risk sets")
  # At risk through the first event, including the event contact.
  data$observed <- 1-data$event
  data <- contact_risk_set(data,"observed",TRUE,FALSE)
  start <- if(is.null(declaration$risk_window_start)) 0 else declaration$risk_window_start
  end <- if(is.null(declaration$risk_window_end)) Inf else declaration$risk_window_end
  data <- data[data$time>start & data$time<=pmin(end,data$observed_time),,drop=FALSE]
  logistic_declaration(data,declaration,"event",ice=TRUE)
}

#' Prepare ICE contacts from released raw participant and event tables.
#' @param participants Raw participant records, including completion_time.
#' @param events Raw event history with event_id, event_kind and occurrence_time.
#' @param contact_times Declared contact schedule in days.
#' @return Canonical contact rows, retaining observed follow-up and discontinuation.
#' @export
ice_contacts <- function(participants,events,contact_times) {
  if(anyDuplicated(participants$participant_id)) stop("duplicate raw participants")
  if(!all(c("participant_id","event_id","event_kind","occurrence_time") %in% names(events))) stop("missing raw ICE event evidence")
  if(!"observed_time" %in% names(participants)) {
    if(!"completion_time" %in% names(participants)) stop("raw follow-up end is required")
    participants$observed_time <- participants$completion_time
  }
  follow <- events[events$event_kind %in% c("endpoint","follow_up"),,drop=FALSE]
  if(nrow(follow)) {
    ends <- tapply(follow$occurrence_time,follow$participant_id,min)
    at <- match(as.character(participants$participant_id),names(ends))
    has <- !is.na(at)
    participants$observed_time[has] <- pmin(participants$observed_time[has],ends[at[has]])
  }
  discontinuation <- events[events$event_id=="treatment_discontinuation",,drop=FALSE]
  first <- tapply(discontinuation$occurrence_time,discontinuation$participant_id,min)
  data <- participants[rep(seq_len(nrow(participants)),each=length(contact_times)),,drop=FALSE]
  data$id <- data$participant_id
  data$time <- rep(contact_times,nrow(participants))
  data$treatment <- as.integer(data$assigned_arm=="active")
  at <- match(as.character(data$id),names(first))
  data$event <- as.integer(!is.na(at) & data$time==first[at])
  data
}

#' Test the declared exponential loss rate by an exact Poisson test.
#' @param data Rows with loss (0/1), and exposure_days or exposure_years through loss/endpoint/admin.
#' @param hazard Declared annual follow-up loss hazard, or NULL.
#' @param process_multiplier Exported censoring hazard multiplier.
#' @param cell_multiplier Exported censoring scale multiplier.
#' @param cluster Whether a declared cluster frailty prevents a marginal rate test.
#' @return A parameter result with the pooled loss count and person-time.
#' @export
loss_rate <- function(data,hazard,process_multiplier=1,cell_multiplier=1,cluster=FALSE) {
  if(cluster) return(parameter_result(list(),"No frailty-free marginal loss rate is declared for DP05/DP06."))
  if(is.null(hazard)) return(parameter_result(list(),"No loss hazard is declared."))
  if("exposure_days" %in% names(data)) data$exposure_years <- data$exposure_days/365.25
  rate <- hazard*process_multiplier/cell_multiplier
  if(!is.finite(rate) || rate<=0 || anyNA(data[c("loss","exposure_years")]) ||
     !all(data$loss %in% 0:1) || any(data$exposure_years<0)) stop("invalid exponential-loss inputs")
  n <- sum(data$loss); exposure <- sum(data$exposure_years)
  if(exposure<=0) stop("loss test needs positive exposure")
  parameter_result(list(list(parameter="loss_rate",numerator=n,exposure_years=exposure,
    realized=n/exposure,declared=rate,p=poisson.test(n,T=exposure,r=rate)$p.value,availability="available")))
}

#' Fit the S08 calendar Cox coefficient.
#' @param data Rows with time,event,intervention,bmi_curve,entry,world,task,cluster.
#' @param offset Declared total calendar log hazard change.
#' @param accrual_days Declared stepped-wedge accrual window.
#' @param profile Optional exported clinical profile for covariate derivation.
#' @return Wald z test of the calendar coefficient.
#' @export
calendar_coefficient <- function(data,offset,accrual_days,profile=NULL) {
  if(!is.finite(accrual_days) || accrual_days<=0) stop("invalid accrual window")
  if(!is.null(profile)) data <- derive_covariates(data,profile)
  required <- c("time","event","intervention","bmi_curve","entry","world","task","cluster")
  if(!all(required %in% names(data)) || anyNA(data[required])) stop("missing calendar-model evidence")
  data$calendar <- data$entry/accrual_days
  data$world_cluster <- interaction(data$world,data$cluster,drop=TRUE)
  tasks <- sort(unique(data$task))
  terms <- paste0("intervention_task_",seq_along(tasks))
  for(i in seq_along(tasks)) data[[terms[i]]] <- data$intervention*as.integer(data$task==tasks[i])
  strata <- survival::strata
  form <- reformulate(c(terms,"bmi_curve","calendar","strata(world)"),"survival::Surv(time,event)")
  fit <- survival::coxph(form,data=data,ties="breslow",cluster=data$world_cluster,
    control=survival::coxph.control(eps=1e-10,iter.max=100,timefix=FALSE))
  if(fit$iter>=100L) stop("calendar Cox did not converge")
  parameter_result(wald_components(fit,c(calendar=offset)))
}

#' TrialDev pool truth extension point.
#' @param ... Reserved for the approved TrialDev contract.
#' @return No result until the TrialDev specification is supplied.
#' @export
pool_truth <- function(...) stop("TrialDev pool_truth specification is pending",call.=FALSE)
