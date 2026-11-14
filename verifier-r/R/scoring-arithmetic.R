# Independent arithmetic for the declared decimal and binary64 reduction contracts.
.score_decimal <- function(x,digits) {
  # mpfr's decimal conversion performs decimal significant-digit rounding with
  # round-to-nearest/even. Re-entering at ample binary precision preserves it.
  bits<-ceiling((digits+20)*log2(10))
  Rmpfr::mpfr(Rmpfr::formatMpfr(x,digits=digits,scientific=TRUE,trim=TRUE),precBits=bits)
}
.score_precision <- function(d) score_rule(d,'arithmetic.metrics_decimal')$precision_decimal_significant_digits
.score_metric <- function(point,truth,se,d) {
  digits<-.score_precision(d);bits<-ceiling((digits+20)*log2(10))
  a<-Rmpfr::mpfr(point,precBits=bits);b<-Rmpfr::mpfr(truth,precBits=bits)
  z<-.score_decimal(a-b,digits)
  Rmpfr::asNumeric(abs(.score_decimal(z/Rmpfr::mpfr(se,precBits=bits),digits)))
}
.score_interval_score <- function(lower,upper,truth,confidence,se,d) {
  digits<-.score_precision(d);bits<-ceiling((digits+20)*log2(10))
  decimal<-function(x).score_decimal(x,digits)
  l<-Rmpfr::mpfr(lower,precBits=bits);u<-Rmpfr::mpfr(upper,precBits=bits)
  t<-Rmpfr::mpfr(truth,precBits=bits);c<-Rmpfr::mpfr(confidence,precBits=bits)
  width<-decimal(u-l);miss<-Rmpfr::pmax(decimal(l-t),decimal(t-u),0)
  score<-decimal(width+decimal(decimal(2*miss)/decimal(1-c)))
  Rmpfr::asNumeric(score)/se
}
.score_fsum <- function(x) {
  if(anyNA(x))return(NA_real_)
  if(!length(x))return(0)
  # An exact binary accumulator needs 2098 bits for the entire binary64 range,
  # plus log2(n) carry bits. One final nearest/even conversion mirrors fsum.
  Rmpfr::asNumeric(sum(Rmpfr::mpfr(x,precBits=2100+ceiling(log2(length(x))))))
}
.score_pysum <- function(x) {
  if(anyNA(x))return(NA_real_)
  s<-0;correction<-0
  for(v in x){t<-s+v;correction<-correction+if(abs(s)>=abs(v))(s-t)+v else (v-t)+s;s<-t}
  s+correction
}
.score_npsum <- function(x) {
  # NumPy's contiguous fast-axis pairwise reduction (8 lanes, block 128).
  n<-length(x)
  if(n<8){s<--0.0;for(v in x)s<-s+v;return(s)}
  if(n<=128) {
    r<-x[1:8];j<-9L
    while(j+7L<=n){r<-r+x[j:(j+7L)];j<-j+8L}
    s<-((r[1]+r[2])+(r[3]+r[4]))+((r[5]+r[6])+(r[7]+r[8]))
    if(j<=n)for(v in x[j:n])s<-s+v
    return(s)
  }
  half<-floor(n/2);half<-half-half %% 8
  .score_npsum(x[seq_len(half)])+.score_npsum(x[seq.int(half+1,n)])
}
.score_npmean <- function(x) .score_npsum(x)/length(x)
.score_npvar <- function(x) .score_npsum((x-.score_npmean(x))^2)/(length(x)-1)
.score_axis_sum <- function(x,axis,column_major=FALSE) {
  contiguous<-if(column_major)0L else 1L
  if(axis==contiguous)return(apply(x,if(axis==0L)2L else 1L,.score_npsum))
  if(axis==0L) {
    s<-numeric(ncol(x));for(j in seq_len(nrow(x)))s<-s+x[j,]
  } else {
    s<-numeric(nrow(x));for(j in seq_len(ncol(x)))s<-s+x[,j]
  }
  s
}
