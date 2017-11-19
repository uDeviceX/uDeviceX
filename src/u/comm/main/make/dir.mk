D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/comm        && \
    d $B/comm/oc     && \
    d $B/d           && \
    d $B/frag        && \
    d $B/glb         && \
    d $B/glb/wvel    && \
    d $B/mpi         && \
    d $B/u/comm/main && \
    d $B/utils      
