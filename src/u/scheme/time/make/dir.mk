D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d             && \
    d $B/mpi           && \
    d $B/scheme/time   && \
    d $B/u/scheme/time && \
    d $B/utils        
