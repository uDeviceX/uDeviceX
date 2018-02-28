D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/mesh_read && \
    d $B/math/tri     && \
    d $B/mpi          && \
    d $B/rbc/matrices && \
    d $B/u/rbc/gen    && \
    d $B/utils       
