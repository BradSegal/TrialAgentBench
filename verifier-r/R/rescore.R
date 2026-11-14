# Independent implementation of the authenticated scoring declaration.
score_trialeval_error <- function(reports,truth,se,measure,declaration,confidence,bank_scale=FALSE) {
  ratio<-identical(measure,'rate_ratio')
  usable<-vapply(reports,function(x) {
    v<-.score_vec(list(x), 'estimate')
    limits<-c(.score_num(x$lower),.score_num(x$upper))
    all(is.finite(c(v,limits))) && (!ratio || all(c(v,limits)>0) || bank_scale)
  },FALSE)
  if(length(truth)==1L)truth<-rep(truth,length(reports))
  if(length(truth)!=length(reports))stop('bank truth and replay lengths differ')
  good<-reports[usable]
  if(!length(good))return(list(own=NA_real_,coverage=NA_real_,share=0,n=0))
  replay_truth<-truth[usable]
  native_truth<-if(ratio)exp(replay_truth) else replay_truth
  comparison_truth<-if(ratio)log(native_truth) else native_truth
  point<-.score_vec(good,'estimate');lower<-.score_vec(good,'lower');upper<-.score_vec(good,'upper')
  if(ratio && bank_scale){point<-exp(point);lower<-exp(lower);upper<-exp(upper)}
  comparison_point<-if(ratio)log(point) else point
  comparison_lower<-if(ratio)log(lower) else lower
  comparison_upper<-if(ratio)log(upper) else upper
  metric<-.score_metric(comparison_point,comparison_truth,se,declaration)
  score<-.score_interval_score(comparison_lower,comparison_upper,comparison_truth,confidence,se,declaration)
  eligible<-is.finite(metric) & is.finite(score)
  errors<-metric[eligible]
  coverage<-as.double(lower[eligible]<=native_truth[eligible] & native_truth[eligible]<=upper[eligible])
  list(own=if(length(errors)).score_fsum(errors)/length(errors) else NA_real_,
    coverage=if(length(coverage)).score_fsum(coverage)/length(coverage) else NA_real_,
    share=length(good)/length(reports),n=length(good))
}
score_trialeval_impute <- function(own,truth,se,measure) {
  observed<-own[is.finite(own)]
  if(!length(observed))stop('empty selected imputation pool')
  no_effect<-abs(if(measure=='rate_ratio')log(truth) else truth)/se
  max(c(observed,no_effect))
}
score_trialeval_complete <- function(own,share,penalty) {
  if(is.na(own))penalty else share*own+(1-share)*penalty
}
score_trialeval_model_effects <- function(values,declaration,column_major=FALSE) {
  mask<-!is.na(values);known<-values;known[!mask]<-0
  nr<-rowSums(mask);nc<-colSums(mask)
  if(any(nr==0) || any(nc==0))stop('empty fit row or column')
  # Advanced column indexing makes the bootstrap input F-contiguous in NumPy.
  colsums<-function(x).score_axis_sum(x,0L,column_major)
  row<-numeric(nrow(values));col<-colsums(known)/nc
  rule<-score_rule(declaration,'trialeval.fit_algorithm')
  for(pass in seq_len(rule$maximum_passes)) {
    adjusted<-sweep(known,2,col,'-')*mask
    updated<-.score_axis_sum(adjusted,1L,column_major)/nr
    col<-colsums(sweep(known,1,updated,'-')*mask)/nc
    delta<-max(abs(updated-row));row<-updated
    if(delta<rule$stop_absolute_threshold)break
  }
  row+.score_npmean(col)
}
score_trialeval_serious_miss <- function(error,reference,declaration) {
  error-reference>=score_rule(declaration,'trialeval.serious_miss')$threshold_SE
}
score_trialdev_study_cost <- function(commitments,enrollments,declaration) {
  costs<-vapply(commitments,function(study) {
    e<-Filter(function(e)e$study_id==study$study_id,enrollments)
    if(length(e)!=1L)stop('study enrollment missing or duplicated')
    screened<-e[[1]]$screened
    if(!is.numeric(screened) || length(screened)!=1L || !is.finite(screened) || screened<0 || screened!=floor(screened))
      stop('screened must be an accrued integer count')
    tariff<-function(key)score_rule(declaration,paste0('trialdev.tariff.',study$phase,'.',key))
    dates<-unique(c(unlist(study$assessment_days),unlist(study$surveillance_days)))
    fixed<-tariff('setup')+study$sites*tariff('per_site')+screened*tariff('screening')
    fixed+length(e[[1]]$patients)*(tariff('enrollment')+study$follow_up_days*tariff('followup_day')+length(dates)*tariff('assessment'))
  },0.0)
  .score_fsum(costs)
}
score_trialdev_approval <- function(active,control,treated_total,declaration) {
  rule<-score_rule(declaration,'trialdev.approval_estimator');weights<-differences<-numeric()
  for(j in seq_along(active)) {
    a<-active[[j]];b<-control[[j]]
    if(min(length(a),length(b))<rule$per_study_n_minimum)next
    variance<-.score_npvar(a)/length(a)+.score_npvar(b)/length(b)
    if(variance<=0)next
    weights<-c(weights,1/variance);differences<-c(differences,.score_npmean(a)-.score_npmean(b))
  }
  z<-if(length(weights))-.score_pysum(weights*differences)/.score_pysum(weights)*sqrt(.score_pysum(weights)) else NA_real_
  gate<-score_rule(declaration,'trialdev.approval_threshold')
  list(z=z,approved=!is.na(z) && z>=gate$critical_z && treated_total>=gate$minimum_completed_treated)
}
score_trialdev_value <- function(approved,outcome,safety,study_cost,day,preferences,declaration) {
  pref<-function(key)if(!is.null(preferences[[key]]))preferences[[key]] else
    score_rule(declaration,paste0('trialdev.preference_default.',key))$default
  clinical<-safety_cost<-0
  if(approved) {
    days<-score_rule(declaration,'trialdev.market_discounting')$year_days
    launch<-(day+pref('regulatory_review_days'))/days;end<-pref('exclusivity_end_day')/days
    rate<-log1p(pref('discount_rate_per_year'))
    years<-if(launch>=end)0 else if(rate==0)end-launch else (exp(-rate*launch)-exp(-rate*end))/rate
    clinical<--outcome/preferences$benefit_unit*pref('annual_profit_per_benefit_unit_usd')*years
    safety_cost<-safety*100*pref('annual_cost_per_safety_percentage_point_usd')*years
  }
  clinical-safety_cost-study_cost-day*pref('programme_cost_per_day_usd')
}
score_share_interval <- function(value,oracle,strata,declaration) {
  rule<-score_rule(declaration,'trialdev.oracle_bootstrap')
  rng<-score_pcg64(rule$seed);groups<-split(seq_along(value),factor(strata,levels=unique(strata)))
  a<-b<-numeric(rule$resamples)
  for(idx in groups) {
    n<-length(idx);if(n<2)return(c(NA_real_,NA_real_))
    pick<-matrix(score_pcg64_integers(rng,rule$resamples*n,n)+1,nrow=rule$resamples,byrow=TRUE)
    mean_a<-.score_npmean(value[idx]);mean_b<-.score_npmean(oracle[idx]);scale<-sqrt(n/(n-1))
    draws_a<-apply(matrix(value[idx][pick],nrow=rule$resamples),1,.score_npmean)
    draws_b<-apply(matrix(oracle[idx][pick],nrow=rule$resamples),1,.score_npmean)
    a<-a+mean_a+scale*(draws_a-mean_a);b<-b+mean_b+scale*(draws_b-mean_b)
  }
  if(any(b<=0))stop('nonpositive bootstrap oracle')
  .score_percentile(a/b,unlist(score_rule(declaration,'bootstrap.quantiles')$percentiles))
}
