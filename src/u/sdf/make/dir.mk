D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/glob         && \
    d $B/io/field     && \
    d $B/io/field/h5  && \
    d $B/io/field/xmf && \
    d $B/math         && \
    d $B/math/rnd     && \
    d $B/mpi          && \
    d $B/sdf          && \
    d $B/sdf/bounce   && \
    d $B/sdf/field    && \
    d $B/sdf/label    && \
    d $B/u/sdf        && \
    d $B/utils        && \
    d $B/wvel        
