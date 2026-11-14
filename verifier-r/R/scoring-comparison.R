# Target diagnostics are used only by the separate comparison process.
score_display_tokens <- function(text) {
  text<-gsub('--',' to ',text,fixed=TRUE)
  text<-gsub('$-$','-',text,fixed=TRUE)
  regmatches(text,gregexpr('-?[0-9]+(?:\\.[0-9]+)?',text,perl=TRUE))[[1]]
}

score_compare_exact <- function(recomputed,stored) {
  if(!is.numeric(recomputed) || !is.character(stored) || length(recomputed)!=length(stored))
    stop('unaligned comparison vectors')
  present<-!is.na(stored) & nzchar(stored)
  if(any(!grepl('^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$',stored[present])))
    stop('stored comparison target is not a numeric token')
  target<-rep(NA_real_,length(stored));target[present]<-as.numeric(stored[present])
  exact<-vapply(seq_along(target),function(j) {
    if(is.na(recomputed[j]) || is.na(target[j]))return(NA)
    identical(writeBin(as.double(recomputed[j]),raw(),size=8),writeBin(as.double(target[j]),raw(),size=8))
  },FALSE)
  near<-abs(recomputed-target)<=1e-12*pmax(1,abs(target))
  data.frame(binary64_equal=exact,relative_1e12=near)
}

score_compare_output <- function(recomputed,target,reason='',display=FALSE,condition='') {
  check<-score_compare_exact(recomputed,target)
  precision<-if(grepl('.',target,fixed=TRUE))nchar(sub('^.*[.]','',target)) else 0L
  display_equal<-display && nzchar(target) && is.finite(recomputed) && round(recomputed,precision)==as.numeric(target)
  status<-if(!nzchar(target))'no_stored_target' else if(nzchar(reason))'not_reproducible' else if(display_equal)'matched_display' else if(isTRUE(check$binary64_equal))'matched_exact' else
    if(isTRUE(check$relative_1e12))'matched_1e-12' else 'mismatched'
  agreement<-status
  if(nzchar(condition) && startsWith(status,'matched'))status<-'conditional'
  list(status=status,agreement_status=agreement,display_equal=display_equal,
    binary64_equal=check$binary64_equal,relative_1e12=check$relative_1e12)
}
