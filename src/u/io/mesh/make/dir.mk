D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf          && \
    d $B/coords        && \
    d $B/d             && \
    d $B/io/mesh       && \
    d $B/io/mesh/write && \
    d $B/io/mesh_read  && \
    d $B/math/tform    && \
    d $B/mesh/vectors  && \
    d $B/mpi           && \
    d $B/u/io/mesh     && \
    d $B/utils         && \
    d $B/utils/string 
