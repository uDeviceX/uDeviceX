D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan   && \
    d $B/d           && \
    d $B/mpi         && \
    d $B/u/algo/scan && \
    d $B/utils      
