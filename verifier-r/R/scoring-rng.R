# PCG XSL RR 128/64 and NumPy's integer SeedSequence expansion, independently
# expressed with exact 16-bit limbs. Algorithm constants (not benchmark knobs):
# https://github.com/numpy/numpy/tree/v2.2.6/numpy/random/src/pcg64
# https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/bit_generator.pyx
.score_xor32 <- function(a,b) {
  as.double(bitwXor(as.integer(a %% 65536),as.integer(b %% 65536))) +
    65536*as.double(bitwXor(as.integer(a %/% 65536),as.integer(b %/% 65536)))
}
.score_mul32 <- function(a,b) {
  ((a %% 65536)*(b %% 65536) +
    (((a %/% 65536)*(b %% 65536)+(b %/% 65536)*(a %% 65536)) %% 65536)*65536) %% 2^32
}
score_seed_sequence <- function(seed) {
  if(is.character(seed)) {
    seed<-unlist(lapply(seed,function(s) {
      z<-gmp::as.bigz(s);base<-gmp::as.bigz(2)^32;w<-numeric()
      if(z<0)stop('invalid integer seed')
      repeat {w<-c(w,as.numeric(z %% base));z<-z %/% base;if(z==0)break};w
    }))
  }
  if (any(!is.finite(seed)) || any(seed<0) || any(seed!=floor(seed))) stop('invalid integer seed')
  entropy <- unlist(lapply(seed,function(s) {
    x <- numeric(); repeat {x<-c(x,s %% 2^32);s<-s %/% 2^32;if(s==0)break};x
  }))
  h <- as.double(0x43b0d7e5)
  hash <- function(x) {
    v <- .score_xor32(x,h); h <<- .score_mul32(h,as.double(0x931e8875))
    v <- .score_mul32(v,h); .score_xor32(v,v %/% 2^16)
  }
  mix <- function(a,b) {
    v <- (.score_mul32(as.double(0xca01f9dd),a)-.score_mul32(as.double(0x4973f715),b)) %% 2^32
    .score_xor32(v,v %/% 2^16)
  }
  pool <- vapply(seq_len(4),function(j)hash(if(j<=length(entropy))entropy[j] else 0),0.0)
  for(s in seq_len(4)) for(t in seq_len(4)) if(s!=t) pool[t]<-mix(pool[t],hash(pool[s]))
  if(length(entropy)>4) for(s in seq.int(5,length(entropy))) for(t in seq_len(4)) pool[t]<-mix(pool[t],hash(entropy[s]))
  h <- as.double(0x8b51f9dd)
  vapply(seq_len(8),function(j) {
    v <- .score_xor32(pool[(j-1) %% 4+1],h); h <<- .score_mul32(h,as.double(0x58f38ded))
    v <- .score_mul32(v,h); .score_xor32(v,v %/% 2^16)
  },0.0)
}
.score_add128 <- function(a,b) {
  r<-a+b;carry<-0
  for(j in seq_len(8)){z<-r[j]+carry;r[j]<-z %% 65536;carry<-z %/% 65536};r
}
.score_step128 <- function(a,increment) {
  # Multiplier 0x2360ed051fc65da44385df649fccf645, little-endian limbs.
  m<-c(63045,40908,57188,17285,23972,8134,60677,9056)
  r<-numeric(8);carry<-0
  for(j in seq_len(8)) {
    z<-sum(a[seq_len(j)]*m[j:1])+increment[j]+carry
    r[j]<-z %% 65536;carry<-z %/% 65536
  };r
}
score_pcg64 <- function(seed) {
  words<-score_seed_sequence(seed)
  limbs<-function(w)as.vector(rbind(w %% 65536,w %/% 65536))
  initial<-limbs(words[c(3,4,1,2)]); inc<-limbs(words[c(7,8,5,6)])
  inc<-.score_add128(inc,inc);inc[1]<-inc[1]+1
  e<-new.env(parent=emptyenv());e$state<-.score_step128(.score_add128(inc,initial),inc)
  e$increment<-inc;e$buffer<-NULL;e
}
score_pcg64_raw <- function(rng,n) {
  result<-matrix(0,n,4)
  for(j in seq_len(n)) {
    s<-.score_step128(rng$state,rng$increment);rng$state<-s
    x<-as.double(bitwXor(as.integer(s[1:4]),as.integer(s[5:8])))
    shift<-s[8] %/% 1024;whole<-shift %/% 16;bits<-shift %% 16
    x<-x[(seq_len(4)+whole-1) %% 4+1]
    if(bits>0) x<-floor(x/2^bits)+(x[c(2,3,4,1)] %% 2^bits)*2^(16-bits)
    result[j,]<-x
  };result
}
score_pcg64_integers <- function(rng,n,high) {
  if(length(high)!=1 || high<1 || high>2^32 || high!=floor(high))stop('uint32 bound required')
  if(high==1)return(rep(0,n))
  result<-numeric(n)
  threshold<-(2^32-high) %% high
  for(j in seq_len(n)) repeat {
    if(is.null(rng$buffer)) {
      x<-score_pcg64_raw(rng,1)[1,];v<-x[1]+65536*x[2];rng$buffer<-x[3]+65536*x[4]
    } else {v<-rng$buffer;rng$buffer<-NULL}
    # Bootstrap bounds are small enough that this product is an exact double.
    if(high>2^21) {
      lo<-(v %% 65536)*high
      mid<-(v %/% 65536)*high+floor(lo/65536)
      lower<-(lo %% 65536)+(mid %% 65536)*65536;upper<-floor(mid/65536)
    } else {m<-v*high;lower<-m %% 2^32;upper<-floor(m/2^32)}
    if(lower>=threshold){result[j]<-upper;break}
  };result
}
score_pcg64_hex <- function(x) apply(x,1,function(r)paste(sprintf('%04x',as.integer(rev(r))),collapse=''))
