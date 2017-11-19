D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm      && \
    d $B/comm/oc   && \
    d $B/d         && \
    d $B/frag      && \
    d $B/mpi       && \
    d $B/u/comm/oc && \
    d $B/utils    
