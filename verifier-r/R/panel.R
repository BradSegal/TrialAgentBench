# Ported from the held R panel; configuration is local to each fit.
panel_functions <- function(cfg) {
z <- if(is.null(cfg$interval_convention) || identical(cfg$interval_df,"groups_minus_one")) qnorm(.975) else
  interval_critical(cfg,n=cfg$n,residual_df=cfg$n-length(cfg$covariates)-2)
normal_result <- function(value, se, critical = z) c(value, se, value - critical * se, value + critical * se)
jackknife <- function(data, point, log_scale = FALSE) {
  value <- point(data)
  groups <- sort(unique(data$delete_group))
  if (length(groups) < 2L) stop("jackknife needs at least two groups")
  replicates <- vapply(groups, function(g) point(data[data$delete_group != g, , drop = FALSE]), numeric(1))
  working <- if (log_scale) log(replicates) else replicates
  se <- sqrt((length(groups) - 1) / length(groups) * sum((working - mean(working))^2))
  critical <- interval_critical(cfg,groups=length(groups),n=nrow(data))
  if (log_scale) c(value, value * se, exp(log(value) - critical * se),
                  exp(log(value) + critical * se)) else normal_result(value,se,critical)
}

endpoint <- function(data) {
  fit <- lm(reformulate(c("treatment",unlist(cfg$covariates)),"y"), data)
  covariance <- sandwich::vcovHC(fit, type = "HC1")
  normal_result(coef(fit)[["treatment"]], sqrt(covariance["treatment","treatment"]),
                interval_critical(cfg,residual_df=df.residual(fit),n=nrow(data)))
}
gee <- function(data) {
  data <- data[order(data$id, data$time), ]
  data$id <- as.integer(factor(data$id))
  data$visit <- factor(data$time)
  form <- if (cfg$time_model == "linear") y ~ treatment * time else y ~ treatment * visit
  # Package-default association moments differ for unbalanced visits. Use
  # packaged conditional mean/sandwich fits and independent matched moments.
  parts <- split(seq_len(nrow(data)), data$id)
  total_pairs <- sum(vapply(parts, function(rows) length(rows) * (length(rows) - 1) / 2, numeric(1)))
  alpha <- 0; previous <- NULL; converged <- FALSE
  for (iteration in seq_len(200L)) {
    fit <- geepack::geeglm(form, id = data$id, data = data, family = gaussian,
                          corstr = "fixed", zcor = rep(alpha, total_pairs),
                          scale.fix = TRUE,
                          control = geepack::geese.control(epsilon = 1e-10, maxit = 200L))
    if (fit$geese$error != 0) stop("conditional GEE did not converge")
    residual <- data$y - fitted(fit); p <- length(coef(fit))
    scale <- sum(residual^2) / (nrow(data) - p)
    cross <- sum(vapply(parts, function(rows)
      (sum(residual[rows])^2 - sum(residual[rows]^2)) / 2, numeric(1)))
    updated <- cross / (scale * (total_pairs - p))
    if (!is.finite(updated) || updated >= 1 || updated <= -1 / (max(lengths(parts)) - 1))
      stop("GEE working correlation is outside positive-definite support")
    if (!is.null(previous) && max(abs(coef(fit) - previous)) < 1e-10 && abs(updated - alpha) < 1e-10) {
      converged <- TRUE; break
    }
    previous <- coef(fit); alpha <- updated
  }
  if (!converged) stop("matched GEE moment iteration did not converge")
  if (fit$geese$error != 0) stop("GEE did not converge")
  at <- data[1, , drop = FALSE]; at$time <- cfg$horizon
  at$visit <- factor(cfg$horizon, levels = levels(data$visit))
  at$treatment <- 1; x1 <- model.matrix(delete.response(terms(fit)), at)
  at$treatment <- 0; x0 <- model.matrix(delete.response(terms(fit)), at)
  contrast <- drop(x1 - x0)
  result <- normal_result(sum(contrast * coef(fit)), sqrt(drop(t(contrast) %*% vcov(fit) %*% contrast)))
  attr(result, "diagnostics") <- list(diagnostic("working_correlation", alpha, "correlation"),
                                     diagnostic("working_scale", scale, "outcome variance"))
  result
}
poisson_contrast <- function(data) {
  columns <- c("treatment", unlist(cfg$covariates, use.names = FALSE))
  fit <- glm(reformulate(columns, "y"), data, family = poisson,
             control = glm.control(epsilon = 1e-12, maxit = 100L))
  if (!fit$converged) stop("Poisson did not converge")
  beta <- coef(fit)[["treatment"]]
  se <- sqrt(sandwich::vcovHC(fit, type = "HC0")["treatment", "treatment"])
  critical <- interval_critical(cfg,residual_df=df.residual(fit),n=nrow(data))
  c(exp(beta), exp(beta) * se, exp(beta - critical * se), exp(beta + critical * se))
}
sequential <- function(data) {
  # Independent reconstruction of stagewise ordering, not gsDesign equivalence.
  information <- head(cfg$information_fractions, nrow(data))
  bounds <- c(data$boundary[-nrow(data)], data$terminal_z[nrow(data)])
  tail_probability <- function(mu) {
    if (length(bounds) == 1L) return(pnorm(bounds - mu, lower.tail = FALSE))
    probability <- gsDesign::gsProbability(k = length(bounds), theta = mu / sqrt(tail(information, 1)),
                                          n.I = information, a = rep(-40, length(bounds)), b = bounds, r = 32)
    sum(probability$upper$prob)
  }
  limits <- vapply(c(.025, .5, .975), function(p)
    uniroot(function(mu) tail_probability(mu) - p,
            c(min((bounds - 12) / sqrt(information / tail(information, 1))), tail(bounds, 1) + 12), tol = 1e-10)$root, numeric(1))
  se <- tail(data$standard_error, 1)
  c(-limits[2] * se, se, -limits[3] * se, -limits[1] * se)
}
# Package KM/Greenwood quantities and independently composed uncertainty.
km_arm <- function(data) {
  time <- pmin(data$time, cfg$horizon)
  event <- data$event == 1 & data$time <= cfg$horizon
  fit <- survfit(Surv(time, event) ~ 1, conf.type = "log-log", timefix = FALSE)
  indices <- which(fit$time <= cfg$horizon)
  survival <- if (length(indices)) tail(fit$surv[indices], 1) else 1
  terms <- ifelse(fit$n.risk > fit$n.event,
                  fit$n.event / (fit$n.risk * (fit$n.risk - fit$n.event)), 0)
  se <- survival * sqrt(sum(terms[indices]))
  knots <- c(0, fit$time[indices], cfg$horizon)
  heights <- c(1, fit$surv[indices])
  areas <- diff(knots) * heights
  rmst <- sum(areas)
  tail_area <- rev(cumsum(rev(areas)))[-1]
  rmst_se <- sqrt(sum(tail_area^2 * terms[indices]))
  risk <- 1 - survival
  if (survival > 0 && survival < 1 && se > 0) {
    v <- se / (survival * abs(log(survival)))
    ci <- c(1 - exp(-exp(log(-log(survival)) - z * v)),
            1 - exp(-exp(log(-log(survival)) + z * v)))
  } else ci <- c(max(0, risk - z * se), min(1, risk + z * se))
  list(risk = risk, se = se, lower = ci[1], upper = ci[2], rmst = rmst, rmst_se = rmst_se)
}
km_contrast <- function(data, rmst = FALSE, normal = FALSE) {
  a <- km_arm(data[data$treatment == 1, ]); c <- km_arm(data[data$treatment == 0, ])
  if (rmst) return(normal_result(a$rmst - c$rmst, sqrt(a$rmst_se^2 + c$rmst_se^2)))
  value <- a$risk - c$risk; se <- sqrt(a$se^2 + c$se^2)
  if (normal) return(normal_result(value, se))
  c(value, se, value - sqrt((a$risk - a$lower)^2 + (c$upper - c$risk)^2),
    value + sqrt((a$upper - a$risk)^2 + (c$risk - c$lower)^2))
}
basis <- function(x, reference, spline) {
  centre <- mean(x); scale <- sd(x)
  train <- (x - centre) / scale; target <- (reference - centre) / scale
  if (!spline) return(list(matrix(train, ncol = 1), matrix(target, ncol = 1)))
  knots <- (quantile(x, c(.05, .35, .65, .95), type = 7) - centre) / scale
  build <- function(values) {
    inner <- knots[3]; last <- knots[4]
    cube <- function(k) pmax(values - k, 0)^3
    cbind(values, sapply(knots[1:2], function(k)
      cube(k) - cube(inner) * (last - k) / (last - inner) +
        cube(last) * (inner - k) / (last - inner)))
  }
  list(build(train), build(target))
}
cox_point <- function(data) {
  target <- data[data$reference_member == 1, , drop = FALSE]
  if (cfg$kind == "cox_risk") {
    design <- matrix(data$treatment, ncol = 1)
    x0 <- matrix(0, nrow(data), 1); x1 <- x0 + 1
  } else {
    if (!nrow(target)) stop("empty empirical target")
    b <- lapply(cfg$covariates, function(name) basis(data[[name]], target[[name]], cfg$basis == "rcs"))
    x <- do.call(cbind, lapply(b, `[[`, 1)); r <- do.call(cbind, lapply(b, `[[`, 2))
    design <- cbind(data$treatment, x, data$treatment * x)
    x0 <- cbind(0, r, 0 * r); x1 <- cbind(1, r, r)
  }
  fit <- coxph(Surv(data$time, data$event) ~ design, ties = "breslow",
               control = coxph.control(eps = 1e-10, iter.max = 100L, timefix = FALSE))
  if (any(!is.finite(coef(fit))) || fit$iter >= 100L) stop("Cox did not converge")
  baseline <- basehaz(fit, centered = FALSE)
  eligible <- baseline$time <= cfg$horizon
  h <- if (any(eligible)) max(baseline$hazard[eligible]) else 0
  mean(exp(-h * exp(drop(x0 %*% coef(fit))))) - mean(exp(-h * exp(drop(x1 %*% coef(fit)))))
}
censor_model <- function(data) {
  x <- scale(as.matrix(data[cfg$covariates]))
  end <- pmin(data$time, cfg$horizon)
  censored <- data$event == 0 & data$time < cfg$horizon - 1e-12
  fit <- coxph(Surv(end, censored) ~ x, ties = "breslow",
               control = coxph.control(eps = 1e-10, iter.max = 100L, timefix = FALSE))
  if (any(!is.finite(coef(fit))) || fit$iter >= 100L) stop("censoring Cox did not converge")
  baseline <- basehaz(fit, centered = FALSE)
  # Survival is nonincreasing. Its minimum on [0,horizon] is at the horizon,
  # for every required baseline profile, even if no endpoint is observed.
  h <- if(nrow(baseline)) max(baseline$hazard[baseline$time<=cfg$horizon],0) else 0
  if(any(exp(-h*exp(drop(x %*% coef(fit)))) < .01))
    stop("fitted censoring survival below declared 0.01 support")
  list(fit = fit, baseline = baseline,
       centre = attr(x, "scaled:center"), scale = attr(x, "scaled:scale"))
}
weighted_curve <- function(data, model) {
  time <- pmin(data$time, cfg$horizon)
  event <- data$event == 1 & data$time <= cfg$horizon + 1e-12
  x <- sweep(sweep(as.matrix(data[cfg$covariates]), 2, model$centre), 2, model$scale, "/")
  hazard <- exp(drop(x %*% coef(model$fit)))
  survival <- 1; area <- 0; previous <- 0
  for (moment in sort(unique(time[event]))) {
    area <- area + (moment - previous) * survival; previous <- moment
    # Strict left limit at censoring ties, events retained in the risk set.
    before <- model$baseline$time < moment
    h <- if (any(before)) max(model$baseline$hazard[before]) else 0
    at_risk <- time >= moment - 1e-12
    failures <- event & abs(time - moment) <= 1e-12
    weights <- exp(h * hazard)
    survival <- survival * (1 - sum(weights[failures]) / sum(weights[at_risk]))
  }
  c(1 - survival, area + (cfg$horizon - previous) * survival)
}
ipcw_point <- function(data, rmst = FALSE, cluster = FALSE) {
  common <- if (!cluster) censor_model(data) else NULL
  arm_value <- function(arm) {
    rows <- data[data$treatment == arm, , drop = FALSE]
    model <- if (cluster) censor_model(rows) else common
    if(!cluster) return(weighted_curve(rows, model))
    clusters <- split(seq_len(nrow(rows)),rows$cluster)
    curves <- vapply(clusters,function(i) weighted_curve(rows[i,,drop=FALSE],model),numeric(2))
    drop(curves %*% (lengths(clusters)/nrow(rows)))
  }
  difference <- arm_value(1) - arm_value(0)
  difference[if (rmst) 2 else 1]
}
stepped_point <- function(data) {
  if (cfg$exposure_strategy != "fixed_at_entry") stop("time-varying stepped-wedge adapter not matched")
  data$period <- floor(data$entry / cfg$period_length)
  data$intervention <- as.numeric(data$entry >= data$switch)
  data$outcome <- as.numeric(data$event == 1 & data$time <= cfg$horizon)
  known <- (data$event == 1 & data$time <= cfg$horizon) | data$time >= cfg$horizon
  weights <- rep(0, nrow(data))
  for (indices in split(seq_len(nrow(data)), interaction(data$period, data$intervention, drop = TRUE))) {
    rows <- data[indices, ]; end <- pmin(rows$time, cfg$horizon)
    censored <- rows$event == 0 & rows$time < cfg$horizon
    fit <- survfit(Surv(end, censored) ~ 1, timefix = FALSE)
    target <- indices[known[indices]]
    before <- findInterval(pmin(data$time[target],cfg$horizon),fit$time,left.open=TRUE)
    probability <- c(1,fit$surv)[before+1L]
    if(any(probability<=0)) stop("no censoring support")
    weights[target] <- 1/probability
  }
  fit <- lm(outcome ~ factor(period) + intervention, data, weights = weights, subset = known)
  if (anyNA(coef(fit))) stop("stepped-wedge design rank deficient")
  unname(coef(fit)[["intervention"]])
}
# Arm-specific saturated multivariate-normal ML, including baseline as a
# response. This is deliberately not a random-intercept/AR(1) REML model.
central_jacobian <- function(fun, parameters) {
  sapply(seq_along(parameters), function(j) {
    step <- 1e-5 * max(1, abs(parameters[j])); up <- down <- parameters
    up[j] <- up[j] + step; down[j] <- down[j] - step
    (fun(up) - fun(down)) / (2 * step)
  })
}
mmrm_arm <- function(data) {
  names <- if (data$treatment[1] == 0 && length(cfg$control_columns)) cfg$control_columns else
    if (data$treatment[1] == 1 && length(cfg$active_columns)) cfg$active_columns else cfg$covariates
  x <- as.matrix(data[names]); n <- nrow(x); p <- ncol(x)
  centre <- colMeans(x, na.rm = TRUE)
  scale <- sqrt(colMeans(sweep(x, 2, centre)^2, na.rm = TRUE))
  x <- sweep(sweep(x, 2, centre), 2, scale, "/")
  long <- data.frame(id = rep(seq_len(n), each = p), visit = rep(seq_len(p), n), y = as.vector(t(x)))
  long$level <- factor(long$visit, levels = seq_len(p))
  fit <- nlme::gls(y ~ level - 1, data = long, method = "ML", na.action = na.omit,
                   correlation = nlme::corSymm(form = ~visit | id),
                   weights = nlme::varIdent(form = ~1 | level),
                   control = nlme::glsControl(msMaxIter = 1000L, tolerance = 1e-10,
                                              msTol = 1e-10, opt = "optim", optimMethod = "BFGS"))
  mu <- unname(coef(fit))
  correlation <- diag(p)
  correlation[lower.tri(correlation)] <- coef(fit$modelStruct$corStruct, unconstrained = FALSE)
  correlation <- correlation + t(correlation) - diag(p)
  variance <- coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  sd <- fit$sigma * variance[as.character(seq_len(p))]
  sigma <- correlation * outer(sd, sd)
  upper <- which(upper.tri(sigma, diag = TRUE), arr.ind = TRUE)
  scores <- function(theta) {
    covariance <- matrix(0, p, p); covariance[upper] <- theta[-seq_len(p)]
    covariance <- covariance + t(covariance) - diag(diag(covariance))
    out <- matrix(0, n, length(theta))
    for (i in seq_len(n)) {
      observed <- which(!is.na(x[i, ]))
      if (!length(observed)) next
      inverse <- matrix(0, p, p)
      inverse[observed, observed] <- solve(covariance[observed, observed, drop = FALSE])
      residual <- x[i, ] - theta[seq_len(p)]; residual[is.na(residual)] <- 0
      weighted <- drop(residual %*% inverse)
      gradient <- outer(weighted, weighted) - inverse
      diag(gradient) <- diag(gradient) / 2
      out[i, ] <- c(weighted, gradient[upper])
    }
    out
  }
  parameters <- c(mu, sigma[upper])
  hessian <- central_jacobian(function(p) colSums(scores(p)), parameters)
  bread <- solve(-(hessian + t(hessian)) / 2)
  # sandwich's generic aggregation, using the full joint observed score.
  object <- structure(list(scores = scores(parameters), bread = n * bread), class = "tab_mmrm")
  covariance <- sandwich::sandwich(object) * n / (n - 1)
  target <- match(cfg$target_column, names)
  c(mu[target] * scale[target] + centre[target], covariance[target, target] * scale[target]^2, n)
}
mmrm <- function(data) {
  a <- mmrm_arm(data[data$treatment == 1, ]); c <- mmrm_arm(data[data$treatment == 0, ])
  # The scored SE is t-to-z equivalent; preserve it as well as the t interval.
  se <- sqrt(a[2] + c[2]) * qt(.975, min(a[3], c[3]) - 1) / z
  normal_result(a[1] - c[1], se)
}
nb2 <- function(data) {
  # glm.nb supplies the packaged ML point fit. The full beta/theta sandwich
  # below is an independent reconstruction, including dispersion uncertainty.
  data$baseline <- as.numeric(scale(data$baseline))
  data$arm <- relevel(factor(data$arm), ref = "control")
  fit <- MASS::glm.nb(y ~ arm + baseline + offset(log(years)), data = data,
                      control = glm.control(epsilon = 1e-10, maxit = 500L), init.theta = 2)
  if (!fit$converged || !is.null(fit$th.warn)) stop("NB2 ML did not converge")
  design <- model.matrix(fit); beta <- coef(fit); theta <- fit$theta
  parameters <- c(beta, log(theta))
  score <- function(parameters) {
    b <- parameters[seq_along(beta)]; size <- exp(tail(parameters, 1))
    mu <- exp(drop(design %*% b)) * data$years
    beta_score <- design * (size * (data$y - mu) / (size + mu))
    size_score <- size * (digamma(data$y + size) - digamma(size) + log(size) + 1 -
                          log(size + mu) - (data$y + size) / (size + mu))
    cbind(beta_score, size_score)
  }
  hessian <- central_jacobian(function(p) colSums(score(p)), parameters)
  bread <- solve(-(hessian + t(hessian)) / 2)
  covariance <- bread %*% crossprod(score(parameters)) %*% bread
  marginal <- function(arm) {
    target <- data; target$arm <- factor(arm, levels = levels(data$arm)); target$years <- 1
    matrix <- model.matrix(delete.response(terms(fit)), target)
    rate <- exp(drop(matrix %*% beta))
    list(value = mean(rate), gradient = colMeans(matrix * rate))
  }
  active <- marginal(cfg$target_arm); control <- marginal("control")
  gradient <- c(active$gradient - control$gradient, 0)
  normal_result(active$value - control$value, sqrt(drop(t(gradient) %*% covariance %*% gradient)))
}
# Independent reconstructions. No packaged clinical estimator is claimed here.
validation <- function(data) {
  if(anyNA(data[c("stratum","selected","y","treatment")]) ||
     !all(data$y %in% 0:1) || !all(data$truth[data$selected] %in% 0:1)) stop("invalid binary likelihood evidence")
  strata <- sort(unique(data$stratum)); s <- length(strata)
  index <- match(data$stratum, strata)
  risk_index <- index + data$treatment * s
  selected <- as.logical(data$selected)
  counts <- tabulate(risk_index, 2 * s)
  if(any(counts==0)) stop("missing arm-stratum support")
  audits <- tabulate(risk_index[selected],2*s)
  if(any(audits<25)) stop("arm-stratum validation support requires at least 25 audited participants")
  boundary <- vapply(strata,function(level) {
    audit <- data[selected & data$stratum==level,,drop=FALSE]
    if(!all(c(0,1) %in% audit$truth)) stop("validation requires audited events and non-events")
    !any(audit$truth==1 & audit$y==0) || !any(audit$truth==0 & audit$y==1)
  },logical(1))
  if (all(selected)) {
    risk <- as.numeric(rowsum(data$truth, risk_index, reorder = TRUE)) / counts
    if(any(!is.finite(risk)) || any(risk<=0 | risk>=1)) stop("boundary validation risks")
    covariance <- diag(risk * (1 - risk) / counts)
  } else {
    # Same bounded observed-data ML criterion and L-BFGS-B algorithm as Python.
    likelihood <- validation_likelihood(data)
    fit <- optim(rep(0, 4 * s), likelihood$nll, gr=likelihood$gradient,
                 method = "L-BFGS-B", lower = -30, upper = 30,
                 control = list(maxit = 2000L, factr = 1e-12 / .Machine$double.eps,
                                pgtol = 1e-8))
    if (fit$convergence != 0 || any(abs(fit$par) >= 30 - 1e-6)) stop("validation ML did not converge")
    p <- plogis(fit$par); risk <- p[seq_len(2 * s)]
    if(any(risk<=0 | risk>=1)) stop("boundary validation risks")
    covariance <- validation_covariance(likelihood$information(fit$par))
  }
  weights <- tabulate(index, s) / nrow(data)
  contrasts <- risk[s + seq_len(s)] - risk[seq_len(s)]
  value <- sum(weights * contrasts)
  gradient <- c(-weights, weights)
  if (!all(selected)) gradient <- c(gradient * risk * (1 - risk), rep(0, 2 * s))
  # The random cohort proportions are part of the current population estimand.
  variance <- drop(t(gradient) %*% covariance %*% gradient) +
    sum(weights * (contrasts - value)^2) / nrow(data)
  result <- normal_result(value, sqrt(variance))
  attr(result,"diagnostics") <- list(diagnostic("accuracy nuisance at probability boundary",any(boundary),"flag"))
  result
}
discrepancy_point <- function(data) {
  means <- vapply(0:1, function(arm) {
    rows <- data[data$treatment == arm, ]
    value <- mean(rows$y)
    for (stratum in sort(unique(rows$stratum))) {
      cell <- rows[rows$stratum == stratum, ]
      audit <- cell[as.logical(cell$selected), ]
      if (!nrow(audit)) stop("empty validation audit cell")
      value <- value + nrow(cell) / nrow(rows) * mean(audit$truth - audit$y)
    }
    value
  }, numeric(1))
  if (cfg$kind == "discrepancy_ratio") {
    if (any(means <= 0)) stop("nonpositive corrected rate")
    means[2] / means[1]
  } else means[2] - means[1]
}
  list(km = km_contrast, rmst = function(d) km_contrast(d, rmst = TRUE),
       safety = function(d) km_contrast(d, normal = TRUE),
       cluster_km = function(d) jackknife(d, function(x) km_contrast(x)[1]),
       cox_risk = function(d) jackknife(d, cox_point),
       cox_standardized = function(d) jackknife(d, cox_point),
       ipcw_risk = function(d) jackknife(d, ipcw_point),
       ipcw_rmst = function(d) jackknife(d, function(x) ipcw_point(x, rmst = TRUE)),
       cluster_ipcw = function(d) jackknife(d, function(x) ipcw_point(x, cluster = TRUE)),
       stepped = function(d) jackknife(d, stepped_point),
       validation = validation, discrepancy = function(d) jackknife(d, discrepancy_point),
       discrepancy_ratio = function(d) jackknife(d, discrepancy_point, log_scale = TRUE),
       endpoint = endpoint, gee = gee, poisson = poisson_contrast,
       sequential = sequential, mmrm = mmrm, nb2 = nb2, binary = binary_contrast)
}
estfun.tab_mmrm <- function(x, ...) x$scores
bread.tab_mmrm <- function(x, ...) x$bread

diagnostic <- function(id, value, unit) {
  list(diagnostic_id = id, value = unname(value), unit = unit)
}

binary_contrast <- function(data) {
  arm <- lapply(0:1, function(a) {
    y <- data$y[data$treatment == a]; n <- length(y); p <- mean(y)
    z <- qnorm(.975); centre <- (p + z^2 / (2*n)) / (1 + z^2/n)
    half <- z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)
    c(p, p*(1-p)/n, centre-half, centre+half)
  })
  a <- arm[[2]]; c <- arm[[1]]; d <- a[1]-c[1]
  c(d, sqrt(a[2]+c[2]), d-sqrt((a[1]-a[3])^2+(c[4]-c[1])^2),
    d+sqrt((a[4]-a[1])^2+(c[1]-c[3])^2))
}
