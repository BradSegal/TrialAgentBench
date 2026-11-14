# Build the auditable registry from exported data, never construction imports.
args <- commandArgs(TRUE)
d <- jsonlite::fromJSON(args[1], simplifyVector=FALSE)
if(identical(d$schema_id,"trialagentbench.public_verification_declarations/v1")) d <- d$declarations
inputs <- jsonlite::fromJSON(if(length(args)>2L) args[3] else file.path(args[2],"route_inputs.json"),simplifyVector=FALSE)
kind_map <- c(coxph_binary_breslow_risk_tau="cox_risk",kaplan_meier_risk_tau="km",
 km_ipcw_rmst_tau="ipcw_rmst",km_rmst_tau="rmst",validated_endpoint_joint_likelihood="validation",
 cox_linear_standardized_risk_tau_reference="cox_standardized",cox_rcs_standardized_risk_tau_reference="cox_standardized",
 validated_event_discrepancy_risk_difference="discrepancy",cluster_parallel_participant_weighted_km="cluster_km",
 cluster_parallel_participant_weighted_km_ipcw_baseline_cox="cluster_ipcw",
 stepped_wedge_period_cluster_adjusted_risk_tau="stepped",group_sequential_adjusted="sequential",
 endpoint_mean_contrast="endpoint",repeated_measures="gee",validation_discrepancy_risk_difference="discrepancy",
 validation_likelihood_risk_difference="validation",poisson_log_link_hc0="poisson",
 validation_discrepancy_rate_ratio="discrepancy_ratio",routine_endpoint_risk_difference="binary_wald",binary_risk_difference="binary")
rows <- list()
for(cell in d$cells) {
 series <- Filter(function(s) s$evaluation_series_id==cell$evaluation_series_id,d$series)[[1]]
 routes <- cell$reference_analyses
 roles <- rep("credited",length(routes))
 if(!is.null(cell$negative_control)) { routes <- c(routes,list(cell$negative_control)); roles <- c(roles,"negative_control") }
 for(i in seq_along(routes)) {
  route <- routes[[i]]; p <- route$procedure; kind <- unname(kind_map[route$estimator_family])
  if(is.na(kind)) stop("unknown declared estimator: ",route$estimator_family)
  covariates <- if(kind %in% c("ipcw_rmst","cluster_ipcw")) c("bmi_centered","bmi_centered_squared") else character()
  if(kind=="cox_standardized") covariates <- unlist(inputs$S05$covariates,use.names=FALSE)
  convention_note <- ""
  if(cell$regime_cell_id=="TE-S17-A2") {
    covariates <- c("age_years","baseline_systolic_blood_pressure_mmhg","baseline_lvef_percent")
    convention_note <- "Operative scoring and released SAP adjust for the three plan baseline covariates; exported procedure text says treatment only."
  }
  endpoint_t <- cell$regime_cell_id %in% c("TE-S10-A1","TE-S11-A1")
  if(endpoint_t) convention_note <- "Operative scoring uses t(n-p), p=2 for this treatment-only model; exported procedure text says normal reference."
  cfg <- list(kind=kind,horizon=series$assessment_horizon_days,covariates=covariates,
    series_id=series$evaluation_series_id,
    endpoint=if(cell$evaluation_series_id %in% c("TE-S04","TE-S06")) "routine_endpoint" else series$primary_endpoint_id,
    endpoint_family=series$endpoint_family,time_model=series$longitudinal_time_model,
    basis=if(grepl("rcs",route$route_id)) "rcs" else "linear",procedure=p)
  cfg$clinical_profile <- series$clinical_profile[c("baseline_fields","prognostic_distribution","prognostic_direction")]
  cfg$clinical_profile$baseline_fields <- lapply(cfg$clinical_profile$baseline_fields,function(f)
    f[c("field_id","source_field_id","offset","scale")])
  if(kind=="cox_standardized") {
    cfg$reference_field <- inputs$S05$reference_field
    cfg$reference_threshold <- inputs$S05$reference_threshold
    cfg$input_provenance <- inputs$S05$provenance
  }
  if(kind=="stepped") {cfg$period_length <- 14;cfg$exposure_strategy <- "fixed_at_entry"}
  set <- if(kind=="sequential") "realized_monitoring_prefix" else
    if(kind %in% c("discrepancy","discrepancy_ratio","validation")) "observed_routine_and_complete_strata" else
    if(kind %in% c("endpoint","gee","poisson","binary","binary_wald")) "observed_routine" else "all_randomized"
  text <- function(x) if(is.null(x)) "" else paste(unlist(x),collapse=";")
  rows[[length(rows)+1]] <- data.frame(cell_id=cell$regime_cell_id,role=roles[i],route_id=route$route_id,
   derivation_status="implemented",derivation_note=if(kind=="cox_standardized") inputs$S05$derivation else "",
   convention_note=convention_note,
   kind=kind,r_function="estimate",formula=switch(kind,poisson=paste("y ~",paste(c("treatment",covariates),collapse=" + ")),endpoint="y ~ treatment",
    gee=if(series$longitudinal_time_model=="linear") "y ~ treatment * time" else "y ~ treatment * factor(time)",
    cox_risk="Surv(time,event) ~ treatment",cox_standardized="Surv(time,event) ~ treatment * basis(covariates)",
    stepped="outcome ~ factor(floor(entry/14)) + intervention",validation="observed-data Bernoulli likelihood",
    cluster_ipcw="Surv(end,censored) ~ bmi_centered + bmi_centered_squared (within arm)",
    ipcw_rmst="Surv(end,censored) ~ bmi_centered + bmi_centered_squared (pooled)",
    "declared arm contrast"),covariates=paste(covariates,collapse=";"),covariates_override="",
   interval_convention=if(endpoint_t) "t" else if(kind=="sequential") "stagewise" else if(p$degrees_of_freedom=="normal_reference") "normal" else "t",
   interval_df=if(endpoint_t) "residual" else if(p$degrees_of_freedom %in% c("normal_reference","sequential_reference")) "" else "groups_minus_one",
   interval_convention_override="",interval_df_override="",analysis_set=set,
   refitting=p$refitting,resampling_count=text(p$resampling_count),resampling_unit=text(p$resampling_unit),
   grouping=p$grouping,degrees_of_freedom=p$degrees_of_freedom,transformation=p$transformation,variance=p$variance,
   fitting_specification=p$fitting_specification,configuration=as.character(jsonlite::toJSON(cfg,auto_unbox=TRUE,digits=17)),
   stringsAsFactors=FALSE)
 }
}
write.csv(do.call(rbind,rows),file.path(args[2],"routes.csv"),row.names=FALSE,na="")
kinds <- c("km","rmst","safety","endpoint","poisson","gee","mmrm","nb2","validation","sequential",
 "cox_risk","cox_standardized","ipcw_risk","ipcw_rmst","cluster_km","cluster_ipcw","stepped",
 "discrepancy","discrepancy_ratio","binary","binary_wald","*")
tolerance <- expand.grid(kind=kinds,field=c("estimate","lower","upper","se"),scale=c("native","ratio"),stringsAsFactors=FALSE)
tolerance$atol <- 2e-8; tolerance$rtol <- 2e-6
for(k in kinds) {
 at <- tolerance$kind==k
 if(k %in% c("km","rmst","safety","endpoint")) {tolerance$atol[at]<-2e-10;tolerance$rtol[at]<-0}
 if(k=="gee") tolerance$atol[at]<-1e-8
 if(k %in% c("mmrm","nb2","validation")) {tolerance$atol[at]<-2e-5;tolerance$rtol[at]<-2e-4}
 if(k=="sequential") tolerance$atol[at]<-2e-6
}
write.csv(tolerance,file.path(args[2],"tolerances.csv"),row.names=FALSE)
