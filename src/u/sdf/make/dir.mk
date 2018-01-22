D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/field     && \
    d $B/io/field/h5  && \
    d $B/io/field/xmf && \
    d $B/math/linal   && \
    d $B/math/rnd     && \
    d $B/math/tform   && \
    d $B/mpi          && \
    d $B/sdf          && \
    d $B/sdf/array3d  && \
    d $B/sdf/bounce   && \
    d $B/sdf/field    && \
    d $B/sdf/label    && \
    d $B/sdf/tex3d    && \
    d $B/sdf/tform    && \
    d $B/u/sdf        && \
    d $B/utils        && \
    d $B/wvel        
