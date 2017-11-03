D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/glb          && \
    d $B/glb/gdot     && \
    d $B/io/field/h5  && \
    d $B/io/field/xmf && \
    d $B/mpi          && \
    d $B/u/h5         && \
    d $B/utils       
