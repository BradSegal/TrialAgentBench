# The authenticated data declaration is the sole owner of benchmark constants.
score_load_declaration <- function(path, expected_sha256) {
  actual <- digest::digest(file = path, algo = "sha256")
  if (!identical(actual, expected_sha256)) stop("scoring declaration SHA256 mismatch")
  d <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  attr(d, "sha256") <- actual
  d
}
score_rule <- function(declaration, name) {
  entry <- declaration$rules[[name]]
  if (is.null(entry)) stop("undeclared convention: ", name)
  entry$value
}
.score_num <- function(x) if (is.null(x) || identical(x, "")) NA_real_ else as.numeric(x)
.score_vec <- function(x, key) vapply(x, function(r) .score_num(r[[key]]), 0.0)
.score_chr <- function(x) if (is.null(x)) "" else as.character(x)
score_write_json <- function(x,path) {
  digits<-ceiling(1+.Machine$double.digits*log10(.Machine$double.base))
  jsonlite::write_json(x,path,auto_unbox=TRUE,null='null',digits=digits)
}
score_write_csv <- function(x, path) {
  x[] <- lapply(x, function(v) if (is.numeric(v)) {
    ifelse(is.na(v), "", sprintf("%.17g", v))
  } else v)
  utils::write.csv(x, path, row.names = FALSE, na = "")
}
