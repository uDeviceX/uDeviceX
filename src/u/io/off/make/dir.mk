D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/mesh_read && \
    d $B/mpi          && \
    d $B/parser       && \
    d $B/u/io/off     && \
    d $B/utils       
