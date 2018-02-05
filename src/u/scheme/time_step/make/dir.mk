D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/force_stat    && \
    d $B/d                  && \
    d $B/io/txt             && \
    d $B/mpi                && \
    d $B/parser             && \
    d $B/u/scheme/time_step && \
    d $B/utils             
