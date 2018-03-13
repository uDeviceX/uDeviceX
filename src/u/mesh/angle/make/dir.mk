D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/kahan_sum && \
    d $B/conf           && \
    d $B/d              && \
    d $B/io/mesh_read   && \
    d $B/math/tri       && \
    d $B/mesh/angle     && \
    d $B/mesh/positions && \
    d $B/mpi            && \
    d $B/u/mesh/angle   && \
    d $B/utils         
