#' Resolve an exported cell route.
#' @param cell_id Declared regime cell identifier.
#' @param route_id Exported route identifier.
#' @param routes_file Route CSV, including optional declaration overrides.
#' @return A configuration list, including the normative procedure.
#' @export
route_configuration <- function(cell_id, route_id, routes_file=system.file("routes.csv", package="tabverify")) {
  routes <- read.csv(routes_file, stringsAsFactors=FALSE, na.strings="NA")
  row <- routes[routes$cell_id==cell_id & routes$route_id==route_id,,drop=FALSE]
  if(nrow(row)!=1L) stop("route is not uniquely declared for this cell")
  cfg <- jsonlite::fromJSON(row$configuration, simplifyVector=TRUE)
  cfg$analysis_set <- row$analysis_set
  choose <- function(name) {
    override <- row[[paste0(name,"_override")]]
    if(length(override) && !is.na(override) && nzchar(override)) override else row[[name]]
  }
  covariates <- choose("covariates")
  cfg$covariates <- if(startsWith(covariates,"[")) unlist(jsonlite::fromJSON(covariates)) else
    if(nzchar(covariates)) strsplit(covariates,";",fixed=TRUE)[[1]] else character()
  if(!is.character(cfg$covariates) && length(cfg$covariates)) stop("covariates must be column identifiers")
  overridden <- row$covariates_override
  cfg$covariates_overridden <- length(overridden)>0L && !is.na(overridden) && nzchar(overridden)
  if(cfg$covariates_overridden) {
    baseline <- cfg$clinical_profile$baseline_fields$field_id
    if(any(!grepl("^[A-Za-z][A-Za-z0-9_]*$",cfg$covariates)) ||
       any(!cfg$covariates %in% baseline) || anyDuplicated(cfg$covariates))
      stop("covariate overrides require declared baseline column identifiers")
  }
  cfg$interval_convention <- choose("interval_convention")
  cfg$interval_df <- choose("interval_df")
  if(identical(row$interval_convention_override,"normal") &&
     (is.null(row$interval_df_override) || is.na(row$interval_df_override) || !nzchar(row$interval_df_override)))
    cfg$interval_df <- ""
  overridden <- c(row$interval_convention_override,row$interval_df_override)
  if(any(!is.na(overridden) & nzchar(overridden))) {
    if(!cfg$interval_convention %in% c("normal","t") ||
       (cfg$interval_convention=="normal" && nzchar(cfg$interval_df)) ||
       (cfg$interval_convention=="t" &&
        (!identical(cfg$interval_df,"residual") || !cfg$kind %in% c("endpoint","poisson"))))
      stop("interval overrides require normal or t with fitted-model residual degrees of freedom")
  }
  cfg
}

interval_critical <- function(cfg, residual_df=NULL, groups=NULL, n=NULL) {
  convention <- cfg$interval_convention
  if(is.null(convention)) {
    if(!is.null(groups)) return(qt(.975,groups-1))
    if(!is.null(residual_df) && is.null(cfg$procedure)) return(qt(.975,residual_df))
    return(qnorm(.975))
  }
  if(convention %in% c("normal","stagewise")) return(qnorm(.975))
  if(convention!="t") stop("unsupported interval convention")
  df <- switch(cfg$interval_df, groups_minus_one=groups-1, residual=residual_df, NULL)
  if(length(df)!=1L || !is.finite(df) || df<=0) stop("invalid interval degrees of freedom")
  qt(.975,df)
}

#' Assign complete-refit deletion groups after analysis-set restriction.
#' @param data Analysis-set data, containing treatment and id or cluster.
#' @param cfg Configuration containing the exported procedure.
#' @return Integer group labels starting at zero.
#' @export
assign_delete_groups <- function(data, cfg) {
  p <- cfg$procedure
  unit <- if(identical(p$resampling_unit,"cluster")) data$cluster else data$id
  if(is.null(unit) || anyNA(unit)) stop("missing independent unit identifiers")
  if(identical(p$resampling_count,"all_independent_units"))
    return(match(unit,sort(unique(unit)))-1L)
  count <- as.integer(p$resampling_count)
  if(length(count)!=1L || is.na(count) || count<2) stop("invalid declared group count")
  labels <- integer(nrow(data))
  for(arm in sort(unique(data$treatment))) {
    rows <- which(data$treatment==arm); units <- unique(as.character(unit[rows]))
    hash <- vapply(units,function(x) digest::digest(enc2utf8(x),algo="sha256",serialize=FALSE),"")
    units <- units[order(hash,method="radix")]
    if(length(units)<count) stop("fewer independent units than declared groups in an arm")
    labels[rows] <- (match(as.character(unit[rows]),units)-1L) %% count
  }
  labels
}

analysis_data <- function(data, cfg, validate_reference=TRUE) {
  if(!is.data.frame(data) || !nrow(data)) stop("empty analysis table")
  if(length(cfg$covariates) && (any(!grepl("^[A-Za-z][A-Za-z0-9_]*$",cfg$covariates)) ||
     any(cfg$covariates %in% c("y","truth","event","treatment","assigned_arm",cfg$endpoint))))
    stop("covariates must be baseline column identifiers")
  if(isTRUE(cfg$covariates_overridden) && !all(cfg$covariates %in% names(data)))
    stop("covariate overrides must exist in the released analysis table")
  # NB2 has multiple named active arms; it owns its arm/exposure preprocessing.
  if(identical(cfg$kind,"nb2")) {
    if(!"arm" %in% names(data) && "assigned_arm" %in% names(data)) data$arm <- data$assigned_arm
    if(!all(c("arm","baseline","years","y") %in% names(data)) ||
       anyNA(data[c("arm","baseline","years","y")]) || any(data$years<=0) ||
       !all(c("control",cfg$target_arm) %in% data$arm)) stop("invalid NB2 analysis inputs")
    return(data)
  }
  if(!"treatment" %in% names(data)) {
    if(!all(data$assigned_arm %in% c("control","active"))) stop("unsupported randomized arms")
    data$treatment <- as.integer(data$assigned_arm=="active")
  }
  if(!all(data$treatment %in% 0:1) || length(unique(data$treatment))!=2L)
    stop("both randomized arms are required")
  aliases <- c(id="participant_id",cluster="cluster_id",time="observed_time",event="event_observed",
    stratum="validation_stratum",selected="validation_selected",truth="adjudicated_endpoint",
    entry="entry_time",switch="switch_time",reference_member="reference_population_member")
  for(name in names(aliases)) if(!name %in% names(data) && aliases[[name]] %in% names(data))
    data[[name]] <- data[[aliases[[name]]]]
  if(!"time" %in% names(data) && "assessment_time" %in% names(data)) data$time <- data$assessment_time
  if(cfg$kind=="gee" && "occurrence_time" %in% names(data)) data$time <- data$occurrence_time
  if(!is.null(cfg$clinical_profile)) data <- derive_covariates(data,cfg$clinical_profile)
  if(!is.null(cfg$bmi_field)) {
    bmi <- (data[[cfg$bmi_field]]-cfg$bmi_offset)/cfg$bmi_scale
    if(length(bmi)!=nrow(data) || anyNA(bmi)) stop("missing pretreatment censoring covariate")
    data$bmi_centered <- bmi-mean(bmi)
    data$bmi_centered_squared <- data$bmi_centered^2
  }
  if(cfg$kind=="cox_standardized") {
    if(!all(c(cfg$covariates,"reference_member",cfg$reference_field) %in% names(data)))
      stop("missing standardization predictors or reference_population_member")
    member <- as.logical(data$reference_member)
    expected <- data[[cfg$reference_field]] >= cfg$reference_threshold
    if(anyNA(member) || anyNA(expected) || any(member!=expected))
      stop("reference_population_member conflicts with protocol BMI threshold")
    data$reference_member <- member
  }
  if(!"y" %in% names(data) && !is.null(cfg$endpoint) && cfg$endpoint %in% names(data)) data$y <- data[[cfg$endpoint]]
  if(!is.null(cfg$analysis_set) && cfg$analysis_set %in% c("observed_routine","observed_routine_and_complete_strata")) {
    if(is.null(data$y)) stop("missing routine endpoint")
    data <- data[!is.na(data$y),,drop=FALSE]
    if(cfg$analysis_set=="observed_routine_and_complete_strata")
      data <- data[!is.na(data$stratum) & nzchar(data$stratum),,drop=FALSE]
  }
  if(validate_reference && cfg$kind %in% c("validation","discrepancy","discrepancy_ratio")) {
    if(is.null(data$selected) || is.null(data$truth)) stop("missing validation evidence")
    data$selected <- as.logical(data$selected)
    if(anyNA(data$selected) || anyNA(data$truth[data$selected])) stop("selected reference endpoints must be observed")
  }
  for(name in intersect(c("y","truth","event"),names(data))) {
    x <- data[[name]]
    if(is.character(x) && all(is.na(x) | x %in% c("true","false","TRUE","FALSE")))
      data[[name]] <- as.integer(as.logical(x))
  }
  data
}

#' Fit a declared analysis or an R-panel configuration.
#' @param data Released analysis table or canonical panel table.
#' @param cfg Configuration from route_configuration, or an explicit panel kind.
#' @param tables Additional CSV tables, including monitoring_plan for sequential fits.
#' @return Named estimate, se, lower, upper vector in native contrast units.
#' @export
estimate <- function(data, cfg, tables=list()) {
  if(!is.null(cfg$unresolved)) stop(cfg$unresolved)
  if(identical(cfg$kind,"sequential")) {
    if(!is.null(tables$monitoring_plan)) {
      plan <- tables$monitoring_plan
      if(nrow(data)>nrow(plan) || !identical(as.character(data$look_id),head(as.character(plan$look_id),nrow(data)))) stop("invalid monitoring prefix")
      cfg$information_fractions <- plan$information_fraction
      boundary <- head(plan$efficacy_z_boundary,nrow(data))
      if(length(boundary)!=nrow(data) || any(!is.finite(boundary))) stop("planned efficacy boundaries are required")
      for(name in intersect(c("boundary","efficacy_z_boundary"),names(data)))
        if(anyNA(data[[name]]) || any(abs(data[[name]]-boundary)>1e-12*pmax(1,abs(boundary))))
          stop("result efficacy boundary conflicts with monitoring plan")
      data$boundary <- boundary
    }
    if(is.null(cfg$information_fractions)) stop("planned information fractions are required")
    if(!"boundary" %in% names(data)) data$boundary <- data$efficacy_z_boundary
    if(!"terminal_z" %in% names(data)) data$terminal_z <- data$z_statistic
  } else {
    data <- analysis_data(data,cfg)
    if(!is.null(cfg$procedure) && cfg$procedure$refitting=="delete_group_jackknife")
      data$delete_group <- assign_delete_groups(data,cfg)
  }
  cfg$n <- nrow(data)
  functions <- panel_functions(cfg)
  if(cfg$kind=="binary_wald") {
    value <- mean(data$y[data$treatment==1])-mean(data$y[data$treatment==0])
    variance <- sum(vapply(0:1,function(a) {
      y <- data$y[data$treatment==a]; mean(y)*(1-mean(y))/length(y)
    },numeric(1)))
    se <- sqrt(variance); result <- c(value,se,value-qnorm(.975)*se,value+qnorm(.975)*se)
  } else {
    fit <- functions[[cfg$kind]]
    if(is.null(fit)) stop("unsupported R configuration: ",cfg$kind)
    result <- fit(data)
  }
  if(length(result)!=4L || any(!is.finite(result)) || result[2]<0 || result[3]>result[4]) stop("invalid reference result")
  names(result) <- c("estimate","se","lower","upper")
  result
}

#' Compare native R values with stored release values.
#' @param result Named estimate vector.
#' @param stored Named estimate/lower/upper values; replay ratios use log units.
#' @param kind R-panel kind.
#' @param ratio Whether this route is a ratio contrast.
#' @param key Whether values come from credited/plan_replay key rows (native units).
#' @return A data frame with both values, tolerance and pass for every field.
#' @export
compare_reference <- function(result, stored, kind, ratio=FALSE, key=FALSE) {
  t <- read.csv(system.file("tolerances.csv",package="tabverify"),stringsAsFactors=FALSE)
  fields <- names(stored); native <- unname(result[fields])
  if(key && kind=="sequential") native <- round(native,6)
  expected <- as.numeric(unlist(stored, use.names=FALSE))
  if(ratio && !key) expected <- exp(expected)
  rows <- t[t$kind==if(kind %in% t$kind) kind else "*",,drop=FALSE]
  rows <- rows[rows$scale==if(ratio) "ratio" else "native",,drop=FALSE]
  rows <- rows[match(fields,rows$field),,drop=FALSE]
  tolerance <- rows$atol+rows$rtol*abs(native)
  data.frame(field=fields,stored=expected,r=native,difference=expected-native,
             tolerance=tolerance,pass=is.finite(native) & abs(expected-native)<=tolerance)
}
