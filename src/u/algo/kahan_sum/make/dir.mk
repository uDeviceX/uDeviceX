D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/kahan_sum   && \
    d $B/d                && \
    d $B/mpi              && \
    d $B/u/algo/kahan_sum && \
    d $B/utils            && \
    d $B/utils/string    
