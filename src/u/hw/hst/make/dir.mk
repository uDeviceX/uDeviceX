D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/mpi          && \
    d $B/u/hw/hst     && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
