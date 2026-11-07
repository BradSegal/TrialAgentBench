"""Real R ingestion preserves released scalar types, identities and missingness."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from trialagentbench_verifier.executor import DEFAULT_EXECUTOR_IMAGE, DockerCodeSession

from trialagentbench.ports.code_execution import AnalysisLanguage, CodeExecutionLimitsV1


def _write_bundled_trial(tmp_path: Path) -> None:
    archive = Path(__file__).parents[2] / "examples/clinical/demo-trialeval-evaluator.zip"
    with ZipFile(archive) as source:
        (member,) = [name for name in source.namelist() if name.endswith("/data/analysis.parquet")]
        (tmp_path / "analysis.parquet").write_bytes(source.read(member))


@pytest.mark.executor
def test_r_retains_analysis_settings_and_package_order_on_produced_data(tmp_path: Path) -> None:
    """A later call must retain the contrast, precision and environment selected for a real trial."""
    _write_bundled_trial(tmp_path)
    session = DockerCodeSession(tmp_path, limits=CodeExecutionLimitsV1())
    try:
        configured = session.execute_result(
            "library(trialagentbenchR); library(MASS); library(survival); "
            "options(digits=17, contrasts=c('contr.sum','contr.poly')); "
            "Sys.setenv(TRIALAGENTBENCH_TEST_SETTING='retained'); "
            "d <- read_trialagentbench_parquet('analysis.parquet'); "
            "d$arm <- factor(d$assigned_arm); "
            "before <- coef(lm(urine_albumin_creatinine_ratio ~ arm, data=d)); "
            "packages_before <- search(); setwd('scratch'); "
            "state_path <- 'participant-owned'; code_path <- 'also-participant-owned'",
            language="r",
        )
        assert configured.status == "success", configured.output
        retained = session.execute_result(
            "stopifnot(getOption('digits')==17, "
            "identical(getOption('contrasts'),c('contr.sum','contr.poly')), "
            "Sys.getenv('TRIALAGENTBENCH_TEST_SETTING')=='retained', "
            "identical(search(),packages_before), basename(getwd())=='scratch', "
            "identical(coef(lm(urine_albumin_creatinine_ratio ~ arm, data=d)),before), "
            "state_path=='participant-owned',code_path=='also-participant-owned'); "
            "cat('produced-trial-session-retained')",
            language="r",
        )
        assert retained.status == "success", retained.output
        assert retained.output == "produced-trial-session-retained"
    finally:
        session.close()


@pytest.mark.executor
def test_r_retains_removed_defaults_and_reordered_packages_on_produced_data(tmp_path: Path) -> None:
    """A new R process must not silently reinstate removed defaults or change function lookup order."""
    _write_bundled_trial(tmp_path)
    session = DockerCodeSession(tmp_path, limits=CodeExecutionLimitsV1())
    try:
        configured = session.execute_result(
            "library(trialagentbenchR); "
            "d<-read_trialagentbench_parquet('analysis.parquet'); stopifnot(nrow(d)>0); "
            "options(contrasts=NULL); detach('package:stats'); packages_before<-search()",
            language="r",
        )
        assert configured.status == "success", configured.output
        retained = session.execute_result(
            "stopifnot(nrow(d)>0,is.null(getOption('contrasts')),identical(search(),packages_before)); "
            "library(stats); packages_before<-search(); contrasts_after_attach<-getOption('contrasts')",
            language="r",
        )
        assert retained.status == "success", retained.output
        reordered = session.execute_result(
            "stopifnot(identical(getOption('contrasts'),contrasts_after_attach),identical(search(),packages_before)); "
            "cat('removed-defaults-and-package-order-retained')",
            language="r",
        )
        assert reordered.status == "success", reordered.output
        assert reordered.output == "removed-defaults-and-package-order-retained"
    finally:
        session.close()


@pytest.mark.executor
def test_r_timeout_while_saving_retains_completed_produced_state(tmp_path: Path) -> None:
    """An interrupted session save must leave the previous real-data analysis state recoverable."""
    _write_bundled_trial(tmp_path)
    session = DockerCodeSession(tmp_path, limits=CodeExecutionLimitsV1(timeout_seconds=6, memory_mb=1024, cpu_count=1))
    try:
        loaded = session.execute_result(
            "d<-read_trialagentbench_parquet('analysis.parquet'); marker<-1; stopifnot(nrow(d)>0)",
            language="r",
        )
        assert loaded.status == "success", loaded.output
        interrupted = session.execute_result(
            "marker<-2; set.seed(121); payload<-runif(30000000); cat('analysis-body-finished')", language="r"
        )
        assert interrupted.status == "timeout", interrupted.output
        assert "analysis-body-finished" in interrupted.output
        recovered = session.execute_result(
            "stopifnot(marker==1,nrow(d)>0); cat('previous-produced-trial-state-restored')", language="r"
        )
        assert recovered.status == "success", recovered.output
        assert recovered.output == "previous-produced-trial-state-restored"
    finally:
        session.close()


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
def test_memory_kill_ends_both_language_sessions_with_produced_data(
    tmp_path: Path, language: AnalysisLanguage
) -> None:
    """A killed interpreter cannot leave a live session serving earlier objects as current results."""
    _write_bundled_trial(tmp_path)
    session = DockerCodeSession(tmp_path, limits=CodeExecutionLimitsV1(timeout_seconds=30, memory_mb=512, cpu_count=1))
    try:
        loaded = session.execute_result(
            (
                "import pandas as pd; d=pd.read_parquet('analysis.parquet'); print(len(d))"
                if language == "python"
                else "library(trialagentbenchR); d<-read_trialagentbench_parquet('analysis.parquet'); cat(nrow(d))"
            ),
            language=language,
        )
        assert loaded.status == "success", loaded.output
        assert int(loaded.output) > 0
        killed = session.execute_result(
            "payload=bytearray(900*1024*1024)" if language == "python" else "payload<-rep(0,128*1024*1024)",
            language=language,
        )
        assert killed.status == "session_terminated", killed.output
        assert "interpreter exited" in killed.output
    finally:
        session.close()


@pytest.mark.executor
def test_r_standard_models_fit_supported_design_controls(tmp_path: Path) -> None:
    """Exercise installed mixed, clustered, survival, count and nonlinear tools."""
    program = """
set.seed(6906)
n <- 300L
d <- data.frame(id=factor(seq_len(n)), site=factor(rep(seq_len(30L), each=10L)),
                arm=rep(0:1, n/2), x=rnorm(n))
u <- rnorm(30L, sd=0.4)
d$y <- 0.4*d$arm + 0.3*d$x + u[d$site] + rnorm(n)
d$count <- rnbinom(n, mu=exp(0.2*d$arm + 0.2*d$x), size=2)
d$time <- rexp(n, exp(-0.3*d$arm + 0.2*d$x))
d$event <- as.integer(d$time < 2)
d$time <- pmin(d$time, 2)
long <- d[rep(seq_len(n), each=3L), ]
long$visit <- rep(0:2, n)
long$y <- long$y + 0.2*long$visit + rnorm(n*3L, sd=0.4)
mixed <- nlme::lme(y ~ arm*visit + x, random=~1|id, data=long)
design <- survey::svydesign(ids=~site, weights=~1, data=d)
clustered <- survey::svyglm(y ~ arm + x, design=design)
survival_fit <- survival::coxph(survival::Surv(time,event) ~ arm + x, cluster=site, data=d)
count_fit <- MASS::glm.nb(count ~ arm + x, data=d)
smooth <- mgcv::gam(y ~ arm + s(x,k=5), data=d, method='REML')
spline <- lm(y ~ arm + splines::ns(x,df=3), data=d)
coefficients <- c(nlme::fixef(mixed), coef(clustered), coef(survival_fit),
                  coef(count_fit), coef(smooth), coef(spline))
standard_errors <- c(sqrt(diag(vcov(mixed))), sqrt(diag(vcov(clustered))),
                    sqrt(diag(vcov(survival_fit))), sqrt(diag(vcov(count_fit))))
stopifnot(all(is.finite(coefficients)), all(is.finite(standard_errors)), all(standard_errors > 0))
cat('mixed-clustered-survival-count-nonlinear-pass')
"""
    (tmp_path / "analysis.R").write_text(program, encoding="utf-8")
    session = DockerCodeSession(tmp_path, image=DEFAULT_EXECUTOR_IMAGE, limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result("source('analysis.R')", language="r")
        assert result.status == "success", result.output
        assert "mixed-clustered-survival-count-nonlinear-pass" in result.output
    finally:
        session.close()


@pytest.mark.executor
def test_r_parquet_preserves_scalar_values_and_missingness(tmp_path: Path) -> None:
    """Typed transfer distinguishes literal NA, empty strings and actual nulls."""
    frame = pd.DataFrame(
        {
            "participant_id": ["001", "NA", "", "__TRIALAGENTBENCH_MISSING__", None, "é,\nquoted"],
            "event_observed": pd.array([True, False, None, False, True, None], dtype="boolean"),
            "count": pd.array([1, 0, None, 17, -2, 10], dtype="Int64"),
            "outcome": [0.123456789012345, 1.0, float("nan"), -0.25, 1e-12, 10.0],
        }
    )
    frame.to_parquet(tmp_path / "input.parquet", index=False)
    session = DockerCodeSession(tmp_path, image=DEFAULT_EXECUTOR_IMAGE, limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); d <- read_trialagentbench_parquet('input.parquet'); "
            "stopifnot(identical(d$participant_id,c('001','NA','','__TRIALAGENTBENCH_MISSING__',NA_character_,'é,\\nquoted')), "
            "identical(d$event_observed,c(TRUE,FALSE,NA,FALSE,TRUE,NA)), "
            "identical(d$count,c(1,0,NA,17,-2,10)), "
            "is.na(d$outcome[3]),abs(d$outcome[1]-0.123456789012345)<1e-16, "
            "d$outcome[5]==1e-12); cat('typed-roundtrip-pass')",
            language="r",
        )
        assert result.status == "success", result.output
        assert "typed-roundtrip-pass" in result.output
    finally:
        session.close()


@pytest.mark.executor
def test_r_parquet_rejects_integer_precision_loss(tmp_path: Path) -> None:
    """A large identity must fail explicitly instead of silently rounding in R."""
    pq.write_table(pa.table({"identity": pa.array([2**53 + 1], type=pa.int64())}), tmp_path / "input.parquet")
    session = DockerCodeSession(tmp_path, image=DEFAULT_EXECUTOR_IMAGE, limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); read_trialagentbench_parquet('input.parquet')", language="r"
        )
        assert result.status == "execution_error"
        assert "lose integer precision" in result.output
    finally:
        session.close()


@pytest.mark.executor
@pytest.mark.parametrize("rows", (0, 3))
def test_r_parquet_preserves_empty_and_null_columns(tmp_path: Path, rows: int) -> None:
    """No observations must not become a fabricated row or inferred text value."""
    pq.write_table(
        pa.table({"id": pa.array([None] * rows, type=pa.string()), "event": pa.array([None] * rows, type=pa.bool_())}),
        tmp_path / "input.parquet",
    )
    session = DockerCodeSession(tmp_path, image=DEFAULT_EXECUTOR_IMAGE, limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); d <- read_trialagentbench_parquet('input.parquet'); "
            f"stopifnot(nrow(d)=={rows},is.character(d$id),is.logical(d$event),all(is.na(d$id)),all(is.na(d$event)))",
            language="r",
        )
        assert result.status == "success", result.output
    finally:
        session.close()
