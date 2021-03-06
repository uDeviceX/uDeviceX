D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/algo/scalars     && \
    d $B/algo/vectors     && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/io/write         && \
    d $B/math/tform       && \
    d $B/mesh/scatter     && \
    d $B/mpi              && \
    d $B/u/mesh/scatter   && \
    d $B/utils            && \
    d $B/utils/nvtx       && \
    d $B/utils/string    
