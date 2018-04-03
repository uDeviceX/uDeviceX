D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/force_stat    && \
    d $B/conf               && \
    d $B/d                  && \
    d $B/io/txt             && \
    d $B/mpi                && \
    d $B/scheme/time_step   && \
    d $B/u/scheme/time_step && \
    d $B/utils              && \
    d $B/utils/nvtx         && \
    d $B/utils/string      
