# Six observable cells per arm/stratum are sufficient for the joint likelihood.
validation_likelihood <- function(data) {
  strata <- sort(unique(data$stratum)); s <- length(strata)
  risk_index <- match(data$stratum,strata) + data$treatment*s
  outcome <- ifelse(data$selected,2+2*data$truth+data$y,data$y)
  code <- (risk_index-1L)*6L+outcome
  cells <- sort(unique(code)); frequencies <- tabulate(match(code,cells),length(cells))
  risk_index <- cells %/% 6L+1L; index <- (risk_index-1L) %% s+1L
  outcome <- cells %% 6L; selected <- outcome>=2L
  routine <- outcome %% 2L; truth <- ifelse(selected,(outcome-2L) %/% 2L,0L)
  # stats::plogis does not accept the complex perturbations used below.
  inverse_logit <- function(theta) 1/(1+exp(-theta))
  nll <- function(theta) {
    p <- inverse_logit(theta); risk <- p[risk_index]
    sensitivity <- p[2*s+index]; specificity <- p[3*s+index]
    marginal <- risk*sensitivity+(1-risk)*(1-specificity)
    audit <- ifelse(truth==1,risk,1-risk) *
      ifelse(truth==1,ifelse(routine==1,sensitivity,1-sensitivity),
             ifelse(routine==0,specificity,1-specificity))
    likelihood <- ifelse(selected,audit,ifelse(routine==1,marginal,1-marginal))
    -sum(frequencies*log(likelihood))
  }
  gradient <- function(theta) {
    p <- inverse_logit(theta); risk <- p[risk_index]
    sensitivity <- p[2*s+index]; specificity <- p[3*s+index]
    marginal <- risk*sensitivity+(1-risk)*(1-specificity)
    score <- (marginal-routine)/(marginal*(1-marginal))
    risk_score <- ifelse(selected,risk-truth,
      score*risk*(1-risk)*(sensitivity+specificity-1))
    sensitivity_score <- ifelse(selected,truth*(sensitivity-routine),
      score*risk*sensitivity*(1-sensitivity))
    specificity_score <- ifelse(selected,(1-truth)*(specificity-(1-routine)),
      -score*(1-risk)*specificity*(1-specificity))
    total <- function(values,groups,n) vapply(seq_len(n),function(k)
      sum(frequencies[groups==k]*values[groups==k]),if(is.complex(theta)) complex(1) else numeric(1))
    c(total(risk_score,risk_index,2*s),total(sensitivity_score,index,s),total(specificity_score,index,s))
  }
  information <- function(theta) {
    # Complex-step differentiation of the analytic score of the full likelihood
    # has no subtraction cancellation, including tiny nuisance curvatures.
    h <- 1e-20
    matrix <- vapply(seq_along(theta),function(k) {
      perturbed <- as.complex(theta); perturbed[k] <- perturbed[k]+1i*h
      Im(gradient(perturbed))/h
    },numeric(length(theta)))
    (matrix+t(matrix))/2
  }
  list(nll=nll,gradient=gradient,information=information)
}

validation_covariance <- function(information) {
  if(any(!is.finite(information)) ||
     min(eigen(information,symmetric=TRUE,only.values=TRUE)$values)<=0) stop("invalid information")
  covariance <- solve(information)
  if(any(!is.finite(covariance))) stop("invalid validation covariance")
  covariance
}
