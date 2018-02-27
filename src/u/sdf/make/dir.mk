D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/field         && \
    d $B/io/field/h5      && \
    d $B/io/field/xmf     && \
    d $B/math/linal       && \
    d $B/math/rnd         && \
    d $B/math/tform       && \
    d $B/math/tri         && \
    d $B/mpi              && \
    d $B/parser           && \
    d $B/u/sdf            && \
    d $B/utils            && \
    d $B/wall/sdf         && \
    d $B/wall/sdf/array3d && \
    d $B/wall/sdf/bounce  && \
    d $B/wall/sdf/field   && \
    d $B/wall/sdf/label   && \
    d $B/wall/sdf/tex3d   && \
    d $B/wall/sdf/tform   && \
    d $B/wall/wvel       
