D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/point     && \
    d $B/mpi          && \
    d $B/u/io/point   && \
    d $B/utils        && \
    d $B/utils/string
