D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/kahan_sum && \
    d $B/conf           && \
    d $B/coords         && \
    d $B/d              && \
    d $B/io/mesh_read   && \
    d $B/math/tform     && \
    d $B/math/tri       && \
    d $B/mesh/area      && \
    d $B/mesh/vectors   && \
    d $B/mesh/volume    && \
    d $B/mpi            && \
    d $B/rbc/gen        && \
    d $B/rbc/matrices   && \
    d $B/u/rbc/gen      && \
    d $B/utils          && \
    d $B/utils/string  
