D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/key_list   && \
    d $B/d               && \
    d $B/mpi             && \
    d $B/u/algo/key_list && \
    d $B/utils           && \
    d $B/utils/nvtx      && \
    d $B/utils/string   
