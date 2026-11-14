score_world_strata <- function(keys) {
  strata<-list()
  series<-sort(unique(vapply(keys,`[[`,'','series_ref')))
  for(s in series) {
    rows<-Filter(function(k)k$series_ref==s,keys)
    cells<-sort(unique(vapply(rows,`[[`,'','cell_ref')))
    worlds<-lapply(cells,function(cell)sort(unique(vapply(Filter(function(k)k$cell_ref==cell,rows),`[[`,'','base_world_id'))))
    same<-all(vapply(worlds,function(w)identical(w,worlds[[1]]),FALSE))
    disjoint<-length(unique(unlist(worlds)))==sum(lengths(worlds))
    if(!same && !disjoint)stop('partially overlapping world strata')
    bundles<-if(same)list(cells) else as.list(cells)
    for(bundle in bundles) {
      w<-worlds[[match(bundle[1],cells)]]
      strata[[length(strata)+1L]]<-lapply(w,function(world) {
        unlist(lapply(bundle,function(cell)sort(vapply(Filter(function(k)k$cell_ref==cell && k$base_world_id==world,rows),`[[`,'','task_id'))))
      })
    }
  }
  n<-unique(lengths(strata));if(length(n)!=1 || n<2)stop('inconsistent finite strata')
  strata
}
score_world_bootstrap <- function(values,keys,declaration) {
  strata<-score_world_strata(keys);n<-length(strata[[1]])
  engine<-score_rule(declaration,'bootstrap.engine');rng<-score_pcg64(engine$seed)
  estimate<-score_trialeval_model_effects(values,declaration)
  draws<-matrix(NA_real_,engine$accepted_resamples,nrow(values))
  accepted<-0L
  column_major<-identical(score_rule(declaration,'trialeval.bootstrap_deviation')$numpy_resample_order,'F')
  while(accepted<nrow(draws)) {
    tasks<-unlist(lapply(strata,function(group)unlist(group[score_pcg64_integers(rng,n,n)+1])))
    sample<-values[,match(tasks,colnames(values)),drop=FALSE]
    if(any(rowSums(!is.na(sample))==0))next
    accepted<-accepted+1L
    draws[accepted,]<-estimate+sqrt(n/(n-1))*(score_trialeval_model_effects(sample,declaration,column_major=column_major)-estimate)
  }
  probs<-unlist(score_rule(declaration,'bootstrap.quantiles')$percentiles)/100
  ci<-t(apply(draws,2,.score_percentile,percentiles=100*probs))
  fixed<-unlist(score_rule(declaration,'trialeval.fit_order')$fixed_rows)
  model_idx<-which(!rownames(values)%in%fixed)
  rank_one<-function(x).score_argsort(.score_argsort(x))
  ranks<-t(apply(draws[,model_idx,drop=FALSE],1,rank_one))
  rank_ci<-t(apply(ranks,2,.score_percentile,percentiles=100*probs))
  list(estimate=estimate,ci=ci,rank=rank_one(estimate[model_idx]),rank_ci=trunc(rank_ci),draws=draws)
}

# NumPy 2.2.6 _lerp: subtract from the upper endpoint when gamma >= 0.5.
# R type 7 is algebraically equivalent but performs different binary64 operations.
.score_percentile <- function(x,percentiles) {
  x<-sort(x);h<-(length(x)-1)*(percentiles/100)
  lo<-floor(h);hi<-pmin(lo+1,length(x)-1);gamma<-h-lo
  a<-x[lo+1];b<-x[hi+1];difference<-b-a
  result<-a+difference*gamma
  upper<-gamma>=0.5
  result[upper]<-b[upper]-difference[upper]*(1-gamma[upper])
  result
}

# NumPy's published generic indexed quicksort: median-of-three partitions,
# insertion sort for <=16 entries, and its equal-key swaps. The declaration
# explicitly records the CPU-dependent equal-key limitation of default argsort.
.score_argsort <- function(x) {
  indices<-seq_along(x)
  swap<-function(a,b){tmp<-indices[a];indices[a]<<-indices[b];indices[b]<<-tmp}
  partition<-function(left,right) {
    if(right<=left)return(invisible(NULL))
    if(right-left<=15) {
      for(j in seq.int(left+1,right)) {
        item<-indices[j];k<-j
        while(k>left && x[item]<x[indices[k-1]]){indices[k]<<-indices[k-1];k<-k-1}
        indices[k]<<-item
      }
      return(invisible(NULL))
    }
    middle<-left+(right-left)%/%2
    if(x[indices[middle]]<x[indices[left]])swap(middle,left)
    if(x[indices[right]]<x[indices[middle]])swap(right,middle)
    if(x[indices[middle]]<x[indices[left]])swap(middle,left)
    pivot<-x[indices[middle]];swap(middle,right-1)
    i<-left;j<-right-1
    repeat {
      repeat{i<-i+1;if(!(x[indices[i]]<pivot))break}
      repeat{j<-j-1;if(!(pivot<x[indices[j]]))break}
      if(i>=j)break
      swap(i,j)
    }
    swap(i,right-1);partition(left,i-1);partition(i+1,right)
    invisible(NULL)
  }
  if(length(x)>1)partition(1,length(x))
  indices
}
