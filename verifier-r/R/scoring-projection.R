# Only facts used by the independent computation cross the preparation boundary.
score_raw_bank <- function(records,fixed_rows) {
  policies<-setdiff(fixed_rows,c('pipeline','policy:scripted_reference'))
  lapply(records,function(r) {
    out<-r[c('task_id','replay_index','truth','reference_route_id')]
    out$routes<-r$routes[unique(c(r$reference_route_id,policies))];out
  })
}

score_oracle_denominators <- function(x) {
  list(values=lapply(x$values,function(v)v[c('task_id','world_id','net_value_usd')]))
}

score_raw_ledger <- function(x) {
  project<-function(x,fields)x[intersect(names(x),fields)]
  out<-project(x,c('resource_day','pool_manifest_sha256'))
  out$terminal<-project(x$terminal,c('decision','programme_day','asset_id','dose'))
  out$recruitment_commitments<-lapply(x$recruitment_commitments,function(s) {
    r<-project(s,c('study_id','phase','sites','follow_up_days','assessment_days','surveillance_days'))
    r$arms<-lapply(s$arms,function(a) {
      list(arm_id=a$arm_id,partition=project(a$partition,c('asset_id','dose','directory')))
    })
    r
  })
  out$enrollments<-lapply(x$enrollments,function(e) {
    r<-project(e,c('study_id','screened'))
    r$patients<-lapply(e$patients,function(p)project(p,c('participant_id','arm_id','enrollment_day')))
    r
  })
  out
}

# These are execution facts, not a stored grading verdict or numeric target.
score_submission_failure <- function(programs,has_result,replay_status,estimate_count) {
  if(!programs)return('invalid_program')
  if(identical(replay_status,'execution_error'))return('execution_error')
  if(identical(replay_status,'reproduced') && estimate_count==0L)return('no_answer')
  if(!has_result && identical(replay_status,'missing_result'))return('missing_result')
  stop('submission evidence does not establish a declared failure')
}
