D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/mpi          && \
    d $B/u/hw/dev     && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
