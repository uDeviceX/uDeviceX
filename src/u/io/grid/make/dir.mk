D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/grid      && \
    d $B/io/grid/h5   && \
    d $B/io/grid/xmf  && \
    d $B/mpi          && \
    d $B/u/io/grid    && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
