D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan && \
    d $B/clistx    && \
    d $B/d         && \
    d $B/mpi       && \
    d $B/u/clist   && \
    d $B/utils    
