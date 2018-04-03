D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/edg         && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/mesh_read     && \
    d $B/io/mesh_read/edg && \
    d $B/mpi              && \
    d $B/u/io/off         && \
    d $B/utils            && \
    d $B/utils/nvtx       && \
    d $B/utils/string    
