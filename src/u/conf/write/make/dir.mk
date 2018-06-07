D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm         && \
    d $B/conf         && \
    d $B/d            && \
    d $B/frag         && \
    d $B/mpi          && \
    d $B/u/conf/write && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
