D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/convert      && \
    d $B/algo/edg          && \
    d $B/algo/force_stat   && \
    d $B/algo/kahan_sum    && \
    d $B/algo/key_list     && \
    d $B/algo/minmax       && \
    d $B/algo/scalars      && \
    d $B/algo/scan         && \
    d $B/algo/vectors      && \
    d $B/clist             && \
    d $B/comm              && \
    d $B/conf              && \
    d $B/coords            && \
    d $B/d                 && \
    d $B/exch/common       && \
    d $B/exch/map          && \
    d $B/exch/mesh         && \
    d $B/flu               && \
    d $B/frag              && \
    d $B/inter/color       && \
    d $B/io/mesh_read      && \
    d $B/io/mesh_read/edg  && \
    d $B/io/restart        && \
    d $B/io/txt            && \
    d $B/math/linal        && \
    d $B/math/rnd          && \
    d $B/math/tform        && \
    d $B/math/tri          && \
    d $B/mesh/bbox         && \
    d $B/mesh/collision    && \
    d $B/mesh/dist         && \
    d $B/mesh/gen          && \
    d $B/mesh/gen/matrices && \
    d $B/mesh/props        && \
    d $B/mesh/triangles    && \
    d $B/mpi               && \
    d $B/rig               && \
    d $B/rig/gen           && \
    d $B/rigid             && \
    d $B/struct/farray     && \
    d $B/struct/parray     && \
    d $B/struct/pfarrays   && \
    d $B/u/rig/gen         && \
    d $B/utils             && \
    d $B/utils/nvtx        && \
    d $B/utils/string     
