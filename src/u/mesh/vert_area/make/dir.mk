D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/io/write         && \
    d $B/math/tform       && \
    d $B/math/tri         && \
    d $B/mesh/vectors     && \
    d $B/mesh/vert_area   && \
    d $B/mpi              && \
    d $B/u/mesh/vert_area && \
    d $B/utils            && \
    d $B/utils/string    
