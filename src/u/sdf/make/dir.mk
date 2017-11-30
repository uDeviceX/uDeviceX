D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/glb          && \
    d $B/glb/wvel     && \
    d $B/io/field     && \
    d $B/io/field/h5  && \
    d $B/io/field/xmf && \
    d $B/mpi          && \
    d $B/sdf          && \
    d $B/u/sdf        && \
    d $B/utils       
