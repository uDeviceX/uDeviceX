D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm         && \
    d $B/d            && \
    d $B/frag         && \
    d $B/math/tri     && \
    d $B/mpi          && \
    d $B/u/math/tri   && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
