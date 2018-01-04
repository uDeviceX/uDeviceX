D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan && \
    d $B/clist     && \
    d $B/d         && \
    d $B/io/off    && \
    d $B/meshbb    && \
    d $B/mpi       && \
    d $B/u/meshbb  && \
    d $B/utils    
