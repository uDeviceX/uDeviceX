D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg          && \
    d $B/algo/kahan_sum    && \
    d $B/algo/key_list     && \
    d $B/algo/scalars      && \
    d $B/algo/vectors      && \
    d $B/conf              && \
    d $B/coords            && \
    d $B/d                 && \
    d $B/io/mesh_read      && \
    d $B/io/mesh_read/edg  && \
    d $B/io/vtk            && \
    d $B/io/vtk/mesh       && \
    d $B/io/write          && \
    d $B/math/tform        && \
    d $B/math/tri          && \
    d $B/mesh/angle        && \
    d $B/mesh/cylindrical  && \
    d $B/mesh/edg_len      && \
    d $B/mesh/eng_kantor   && \
    d $B/mesh/scatter      && \
    d $B/mesh/vert_area    && \
    d $B/mpi               && \
    d $B/u/mesh/eng_kantor && \
    d $B/utils             && \
    d $B/utils/nvtx        && \
    d $B/utils/string     
