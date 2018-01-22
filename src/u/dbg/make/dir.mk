D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d        && \
    d $B/dbg      && \
    d $B/glob     && \
    d $B/io/punto && \
    d $B/mpi      && \
    d $B/parser   && \
    d $B/u/dbg    && \
    d $B/utils   
