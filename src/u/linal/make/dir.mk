D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm       && \
    d $B/d          && \
    d $B/frag       && \
    d $B/math/linal && \
    d $B/math/rnd   && \
    d $B/mpi        && \
    d $B/u/linal    && \
    d $B/utils     
