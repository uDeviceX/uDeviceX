D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d              && \
    d $B/io/mesh_read   && \
    d $B/mesh/positions && \
    d $B/mesh/volume    && \
    d $B/mpi            && \
    d $B/parser         && \
    d $B/u/mesh/volume  && \
    d $B/utils         
