D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/force_stat   && \
    d $B/conf              && \
    d $B/d                 && \
    d $B/io/txt            && \
    d $B/mpi               && \
    d $B/u/algo/force_stat && \
    d $B/utils            
