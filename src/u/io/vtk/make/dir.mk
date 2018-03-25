D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/algo/kahan_sum   && \
    d $B/algo/key_list    && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/io/vtk           && \
    d $B/io/vtk/mesh      && \
    d $B/math/tform       && \
    d $B/math/tri         && \
    d $B/mesh/tri_area    && \
    d $B/mesh/vectors     && \
    d $B/mpi              && \
    d $B/u/io/vtk         && \
    d $B/utils            && \
    d $B/utils/string    
