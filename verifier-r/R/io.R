# JSON names follow the property-verification request and response contracts.
protocol_fields <- list(
  request=c("type","cell_id","route_id","tables","configuration","parameters"),
  estimate=c("type","status","estimate","lower","upper","se","message","versions"),
  mechanism=c("type","p_diff","components","availability","message"),
  pool_truth=c("type","contrasts")
)

#' Read a request directory containing request.json and CSV tables.
#' @param dir Request directory.
#' @return Validated request list and decoded tables.
#' @export
read_payload <- function(dir) {
  root <- normalizePath(dir,mustWork=TRUE)
  request <- jsonlite::fromJSON(file.path(root,"request.json"),simplifyVector=FALSE)
  if(!is.list(request) || is.null(request$type) || length(request$type)!=1L) stop("request type required")
  if(length(setdiff(names(request),protocol_fields$request))) stop("unknown request JSON fields")
  if(!request$type %in% c("estimate","mechanism","mechanism_parameters","observation_coefficients","ice_coefficients","loss_rate","calendar_coefficient","pool_truth")) stop("unsupported request type")
  if(!is.list(request$tables) || !length(request$tables) || is.null(names(request$tables))) stop("named CSV tables required")
  request$tables <- lapply(request$tables,function(relative) {
    if(!is.character(relative) || length(relative)!=1 || grepl("^/|(^|/)\\.\\.(/|$)",relative)) stop("table must be a request-relative CSV path")
    path <- normalizePath(file.path(root,relative),mustWork=TRUE)
    if(!startsWith(path,paste0(root,"/")) || !grepl("\\.csv$",path)) stop("table lies outside request directory or is not CSV")
    data <- read.csv(path,stringsAsFactors=FALSE,check.names=FALSE,na.strings=c("","NA"))
    for(name in names(data)) {
      x <- data[[name]]
      if(is.character(x) && any(!is.na(x)) && all(is.na(x) | x %in% c("true","false","TRUE","FALSE")))
        data[[name]] <- as.logical(x)
    }
    data
  })
  request
}

package_versions <- function() {
  packages <- c("tabverify","digest","geepack","gsDesign","jsonlite","MASS","nlme","sandwich","survival")
  c(list(R=as.character(getRversion())),setNames(lapply(packages,function(p) as.character(utils::packageVersion(p))),packages))
}

#' Write a protocol response.
#' @param path Output JSON path.
#' @param result RResult value.
#' @return Invisibly, the serialized result.
#' @export
write_result <- function(path,result) {
  fields <- protocol_fields[[result$type]]
  if(is.null(fields) || length(setdiff(names(result),fields)) || length(setdiff(setdiff(fields,c("se","p_diff")),names(result)))) stop("invalid result fields")
  if(result$type=="estimate") {
    if(!result$status %in% c("ok","failed","unavailable")) stop("invalid estimate status")
    numeric_fields <- c("estimate","lower","upper")
    if(!is.null(result$se)) numeric_fields <- c(numeric_fields,"se")
    if(result$status=="ok" && (any(!vapply(result[numeric_fields],function(x) is.numeric(x) && length(x)==1 && is.finite(x),logical(1))) || result$lower>result$upper || (!is.null(result$se) && result$se<0))) stop("invalid estimate values")
  }
  jsonlite::write_json(result,path,auto_unbox=TRUE,digits=17,pretty=TRUE,null="null",na="null")
  invisible(result)
}

#' Execute a request and write response.json and sessionInfo.txt.
#' @param dir Request directory.
#' @param out Output directory.
#' @return Invisibly, the protocol result.
#' @export
run <- function(dir,out) {
  request <- read_payload(dir)
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  capture.output(utils::sessionInfo(),file=file.path(out,"sessionInfo.txt"))
  tables <- request$tables
  data <- tables$data
  if(is.null(data)) stop("a table named data is required")
  p <- request$parameters
  if(is.null(p)) p <- list()
  if(request$type=="estimate") {
    cfg <- request$configuration
    if(!is.null(request$cell_id) || !is.null(request$route_id)) {
      if(is.null(request$cell_id) || is.null(request$route_id) || !is.null(cfg)) stop("supply a cell/route pair or a panel configuration")
      cfg <- route_configuration(request$cell_id,request$route_id)
    }
    if(is.null(cfg$kind)) stop("estimate configuration required")
    result <- tryCatch({
      value <- estimate(data,cfg,tables)
      flags <- Filter(function(d)identical(d$unit,"flag") && isTRUE(d$value),attr(value,"diagnostics"))
      message <- paste(vapply(flags,`[[`,"","diagnostic_id"),collapse="; ")
      list(type="estimate",status="ok",estimate=unname(value["estimate"]),lower=unname(value["lower"]),upper=unname(value["upper"]),se=unname(value["se"]),message=message,versions=package_versions())
    },error=function(e) list(type="estimate",status="failed",estimate=NULL,lower=NULL,upper=NULL,se=NULL,message=conditionMessage(e),versions=package_versions()))
  } else {
    fun <- switch(request$type,mechanism=mechanism,mechanism_parameters=mechanism_parameters,
      observation_coefficients=observation_coefficients,ice_coefficients=ice_coefficients,
      loss_rate=loss_rate,calendar_coefficient=calendar_coefficient,pool_truth=pool_truth)
    result <- do.call(fun,c(list(data=data),p))
  }
  write_result(file.path(out,"response.json"),result)
}
