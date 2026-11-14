#' Derive model covariates from the exported clinical profile.
#' @param data Released baseline fields, possibly repeated over contacts.
#' @param profile Clinical profile with baseline_fields and prognostic constants.
#' @return Data with canonical source fields, bmi_curve and IPCW basis columns.
#' @export
derive_covariates <- function(data,profile) {
  fields <- profile$baseline_fields
  if(is.data.frame(fields)) fields <- lapply(seq_len(nrow(fields)),function(i) fields[i,,drop=FALSE])
  prognostic <- NULL
  for(field in fields) {
    if(!field$field_id %in% names(data)) next
    if(!is.null(field$source_field_id) && is.numeric(data[[field$field_id]])) {
      if(!is.finite(field$scale) || field$scale==0) stop("invalid baseline mapping scale")
      data[[field$source_field_id]] <- (data[[field$field_id]]-field$offset)/field$scale
      if(field$source_field_id=="bmi") prognostic <- data[[field$field_id]]
    }
  }
  if(!is.null(prognostic)) {
    distribution <- profile$prognostic_distribution
    if(!is.finite(distribution$standard_deviation) || distribution$standard_deviation<=0 ||
       !is.finite(distribution$mean) || !profile$prognostic_direction %in% c(-1,1))
      stop("invalid declared prognostic constants")
    # Population constants, signed linear effect with intercept zero and power one.
    data$bmi_curve <- profile$prognostic_direction *
      (prognostic-distribution$mean)/distribution$standard_deviation
    data$bmi_centered <- data$bmi-mean(data$bmi)
    data$bmi_centered_squared <- data$bmi_centered^2
  }
  data
}
