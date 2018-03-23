D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm         && \
    d $B/d            && \
    d $B/frag         && \
    d $B/math/linal   && \
    d $B/math/rnd     && \
    d $B/math/tform   && \
    d $B/math/tri     && \
    d $B/mpi          && \
    d $B/u/math/linal && \
    d $B/utils        && \
    d $B/utils/string
