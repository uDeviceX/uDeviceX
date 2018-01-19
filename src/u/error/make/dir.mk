D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm    && \
    d $B/d       && \
    d $B/frag    && \
    d $B/mpi     && \
    d $B/parser  && \
    d $B/u/error && \
    d $B/utils  
