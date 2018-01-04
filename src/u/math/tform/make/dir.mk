D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm             && \
    d $B/d                && \
    d $B/frag             && \
    d $B/glob             && \
    d $B/math/linal       && \
    d $B/math/rnd         && \
    d $B/math/tform       && \
    d $B/mpi              && \
    d $B/u/math/tform     && \
    d $B/u/math/tform/lib && \
    d $B/utils           
