D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/mpi          && \
    d $B/u/io/h5      && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
