D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf             && \
    d $B/coords           && \
    d $B/d                && \
    d $B/io/restart       && \
    d $B/mpi              && \
    d $B/scheme/time_line && \
    d $B/u/scheme/time    && \
    d $B/utils            && \
    d $B/utils/convert    && \
    d $B/utils/nvtx       && \
    d $B/utils/string    
