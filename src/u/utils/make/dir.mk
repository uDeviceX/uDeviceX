D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d       && \
    d $B/mpi     && \
    d $B/u/utils && \
    d $B/utils  