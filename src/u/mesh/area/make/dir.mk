D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/kahan_sum && \
    d $B/d              && \
    d $B/io/mesh_read   && \
    d $B/math/tri       && \
    d $B/mesh/area      && \
    d $B/mesh/positions && \
    d $B/mpi            && \
    d $B/parser         && \
    d $B/u/mesh/area    && \
    d $B/utils         
