D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm             && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/frag             && \
    d $B/math/linal       && \
    d $B/math/rnd         && \
    d $B/math/tform       && \
    d $B/math/tri         && \
    d $B/mpi              && \
    d $B/u/math/tform     && \
    d $B/u/math/tform/lib && \
    d $B/utils            && \
    d $B/wall/sdf/tform  
