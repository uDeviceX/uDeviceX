D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/dbg          && \
    d $B/io/txt       && \
    d $B/mpi          && \
    d $B/u/dbg        && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
