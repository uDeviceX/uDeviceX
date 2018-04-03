D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan    && \
    d $B/clist        && \
    d $B/conf         && \
    d $B/coords       && \
    d $B/d            && \
    d $B/mpi          && \
    d $B/u/clist      && \
    d $B/utils        && \
    d $B/utils/nvtx   && \
    d $B/utils/string
