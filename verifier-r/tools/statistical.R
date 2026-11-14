# Statistical orchestration; tabverify owns every estimator and process model.
args <- commandArgs(TRUE)
library(tabverify)
job <- jsonlite::fromJSON(args[1],simplifyVector=FALSE)
write <- function(value,path) jsonlite::write_json(value,path,auto_unbox=TRUE,digits=17,pretty=TRUE,null="null",na="null")
pooled <- ice <- raw <- list()
read <- function(path) {
 d <- read.csv(path,stringsAsFactors=FALSE,check.names=FALSE,na.strings=c("","NA"))
 for(n in names(d)) if(is.character(d[[n]]) && any(!is.na(d[[n]])) && all(is.na(d[[n]]) | d[[n]] %in% c("True","False","true","false","TRUE","FALSE"))) d[[n]] <- tolower(d[[n]])=="true"
 d
}
requests <- unlist(lapply(job$worlds,`[[`,"requests"),use.names=FALSE)
if(length(args)<2L || args[2]!="parameters-only") {
 completed <- parallel::mclapply(requests,function(request) {
  tabverify::run(request,request)
  TRUE
 },mc.cores=job$fit_workers,mc.set.seed=FALSE)
 if(!all(vapply(completed,isTRUE,logical(1)))) stop("R request worker failed; retained responses are incomplete")
}
for(world in job$worlds) {
 d <- read(world$analysis)
 cfg <- route_configuration(job$cell$regime_cell_id,job$cell$reference_analyses[[1]]$route_id)
 cfg$analysis_set <- "all"
 # Process checks use the full cohort, including legitimately missing reference
 # endpoints. Fit-specific reference support remains enforced by every estimator.
 d <- tabverify:::analysis_data(d,cfg,validate_reference=FALSE)
 d$id <- paste(world$task_id,world$replication,d$id,sep=":")
 d$world <- paste(world$task_id,world$replication,sep=":")
 d$task <- world$task_id
 if("entry" %in% names(d) && "switch" %in% names(d)) d$intervention <- as.integer(d$entry>=d$switch)
 pooled[[length(pooled)+1L]] <- d
 if(length(world$raw)) {
  tables <- lapply(world$raw,read)
  for(n in names(tables)) if("participant_id" %in% names(tables[[n]])) tables[[n]]$participant_id <- paste(world$task_id,world$replication,tables[[n]]$participant_id,sep=":")
  raw[[length(raw)+1L]] <- tables
  if(length(job$cell$intercurrent_events)) ice[[length(ice)+1L]] <- ice_contacts(tables$participant_records,tables$event_history,unlist(job$profile$contact_times_days))
 }
}
d <- do.call(rbind,pooled)
results <- list()
evaluate <- function(id,field,fn) {
 # An estimation error is an explicit unavailable finding, with its cause.
 result <- tryCatch(fn(),error=function(e) list(type="mechanism",p_diff=NULL,components=list(),availability="unavailable",message=conditionMessage(e)))
 for(i in seq_along(result$components)) {
  component <- result$components[[i]]
  if(is.null(component$declaration_field)) {
   parameter <- component$parameter
   if(id=="L1-MECH") {
    stratum <- Filter(function(s)identical(s$value,component$stratum),cell$endpoint_validation_strata)[[1]]
    prefix <- paste0(field,"[value=",component$stratum,"].")
    suffix <- if(parameter=="selection") "validation_probability" else if(parameter=="capture") "capture_probability" else parameter
    if(!is.null(component$arm)) {
     by_arm <- if(parameter=="capture") "capture_by_arm" else "accuracy_by_arm"
     if(length(stratum[[by_arm]])) prefix <- paste0(prefix,by_arm,"[arm_id=",component$arm,"].")
    }
   } else {
    prefix <- paste0(field,".")
    suffix <- switch(parameter,"(Intercept)"="intercept",time="time_coefficient",bmi_curve="baseline_coefficient",treatment="arm_coefficients[active].value - arm_coefficients[control].value",calendar="calendar_final_log_hazard_offset",loss_rate="follow_up_loss_hazard_per_year * process_multiplier / cell_multiplier",paste0("effects[input_variable=",parameter,"].coefficient"))
    if(id=="L1-ICE" && parameter=="(Intercept)") suffix <- "intercept + arm_coefficients[control].value"
   }
   result$components[[i]]$declaration_field <- paste0(prefix,suffix)
  }
 }
 results[[length(results)+1L]] <<- list(check_id=id,declaration_field=field,result=result,n=nrow(d))
}
cell <- job$cell; series <- job$series; profile <- job$profile
if(length(cell$endpoint_validation_strata)) evaluate("L1-MECH","endpoint_validation_strata",function() mechanism_parameters(d,cell$endpoint_validation_strata,if(series$endpoint_family=="count") "count" else "binary",recorded_domain=TRUE))
if(series$evaluation_series_id %in% paste0("TE-S",10:13)) {
 evaluate("L1-OBS","realism_observation_layer",function() {
  declaration <- read_observation_declaration(job$realism_path,job$realism_sha256,series$evaluation_series_id)
  canonical <- lapply(raw,function(tables) tabverify:::released_observation_data(tables,declaration))
  continuous <- declaration$mechanism=="continuous_time_withdrawal_plus_intermittent_measurement"
  data <- if(continuous) list(participants=do.call(rbind,lapply(canonical,`[[`,"participants")),contacts=do.call(rbind,lapply(canonical,`[[`,"contacts"))) else do.call(rbind,canonical)
  observation_coefficients(data,declaration,series_id=series$evaluation_series_id)
 })
} else if(!is.null(cell$observation)) {
 evaluate("L1-OBS","observation",function() {
  if("occurrence_time" %in% names(d)) d$time <- d$occurrence_time
  if(!"observed" %in% names(d)) stop("released contact observation indicators unavailable")
  d$terminal <- d$time==max(unlist(profile$contact_times_days))
  observation_coefficients(d,cell$observation,profile=series$clinical_profile,series_id=series$evaluation_series_id)
 })
}
if(length(cell$intercurrent_events)) for(event in cell$intercurrent_events) evaluate("L1-ICE",paste0("intercurrent_events[event_id=",event$event_id,"]"),function() ice_coefficients(do.call(rbind,ice),event,profile=series$clinical_profile))
evaluate("L1-CENS","follow_up_loss_hazard_per_year * censoring_hazard_multiplier / censoring_scale_multiplier",function() {
 if(!is.null(series$follow_up_loss_hazard_per_year) && !all(c("event_type","observed_time") %in% names(d))) {
  if(cell$regime_cell_id %in% c("TE-S04-A2","TE-S06-A2","TE-S06-A3")) stop(paste(
   "The released ADSL and DS disposition in standards.zip does not identify the declared exponential loss.",
   "In S01, DS DISCONTINUED (3,746) and missed final contact (2,137) do not match the 3,991 losses.",
   "In S06, missed final contact is 3.7%, against at most 1.5% possible under the declared hazard."))
  stop("released loss indicator and exposure unavailable")
 }
 loss_rate(if(all(c("event_type","observed_time") %in% names(d))) data.frame(loss=as.integer(d$event_type=="censoring"),exposure_days=d$observed_time) else NULL,
 series$follow_up_loss_hazard_per_year,profile$censoring_hazard_multiplier,cell$censoring_scale_multiplier,cluster=job$cluster)
})
if(identical(series$evaluation_series_id,"TE-S08")) {
 evaluate("L1-CAL","calendar_final_log_hazard_offset",function() calendar_coefficient(d,cell$calendar_final_log_hazard_offset,profile$stepped_wedge_accrual_days,profile=series$clinical_profile))
} else {
 evaluate("L1-CAL","calendar_final_log_hazard_offset",function() tabverify:::parameter_result(list(),"No calendar effect is active outside S08."))
}
write(results,job$result)
capture.output(sessionInfo(),file=file.path(dirname(job$result),"sessionInfo.txt"))
