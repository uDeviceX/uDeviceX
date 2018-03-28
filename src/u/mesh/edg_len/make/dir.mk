D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/math/tform       && \
    d $B/mesh/edg_len     && \
    d $B/algo/vectors     && \
    d $B/mpi              && \
    d $B/u/mesh/edg_len   && \
    d $B/utils            && \
    d $B/utils/string    
