D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d           && \
    d $B/glb         && \
    d $B/glb/gdot    && \
    d $B/io/field/h5 && \
    d $B/mpi         && \
    d $B/u/hdf       && \
    d $B/utils      
