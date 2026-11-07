#' Read participant-visible Parquet data
#'
#' Converts one Parquet file through the executor's pinned PyArrow dependency
#' and returns a base R data frame. Conversion occurs only in the container's
#' bounded temporary filesystem and never changes the source file.
#'
#' @param path Relative or absolute path to a Parquet file under the task workspace.
#' @return A data frame with source column names preserved.
#' @export
read_trialagentbench_parquet <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("path must be one non-empty string", call. = FALSE)
  }
  source <- normalizePath(path, mustWork = TRUE)
  output <- tempfile(pattern = "trialagentbench-r-", fileext = ".csv", tmpdir = "/tmp")
  on.exit(unlink(output), add = TRUE)
  prior_input <- Sys.getenv("TRIALAGENTBENCH_R_PARQUET_INPUT", unset = NA_character_)
  prior_output <- Sys.getenv("TRIALAGENTBENCH_R_CSV_OUTPUT", unset = NA_character_)
  on.exit({
    if (is.na(prior_input)) Sys.unsetenv("TRIALAGENTBENCH_R_PARQUET_INPUT") else Sys.setenv(TRIALAGENTBENCH_R_PARQUET_INPUT = prior_input)
    if (is.na(prior_output)) Sys.unsetenv("TRIALAGENTBENCH_R_CSV_OUTPUT") else Sys.setenv(TRIALAGENTBENCH_R_CSV_OUTPUT = prior_output)
  }, add = TRUE)
  Sys.setenv(TRIALAGENTBENCH_R_PARQUET_INPUT = source, TRIALAGENTBENCH_R_CSV_OUTPUT = output)
  conversion <- system2(
    "python",
    "/opt/trialagentbench/parquet_to_csv.py",
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(conversion, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(c("Parquet conversion failed.", conversion), collapse = "\n"), call. = FALSE)
  }
  schema <- jsonlite::fromJSON(readLines(output, n = 1L, warn = FALSE))
  result <- utils::read.csv(
    output, skip = 1L, check.names = FALSE, stringsAsFactors = FALSE,
    colClasses = schema$classes, na.strings = schema$missing,
    strip.white = FALSE, comment.char = "", fileEncoding = "UTF-8"
  )
  if (!identical(names(result), schema$columns)) {
    stop("Parquet conversion changed column identities.", call. = FALSE)
  }
  result
}
