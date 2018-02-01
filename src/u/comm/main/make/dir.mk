D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm        && \
    d $B/coords      && \
    d $B/d           && \
    d $B/frag        && \
    d $B/mpi         && \
    d $B/parser      && \
    d $B/u/comm/main && \
    d $B/utils      
