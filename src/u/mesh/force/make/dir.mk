D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d                  && \
    d $B/mesh/force/kantor0 && \
    d $B/mesh/force/kantor1 && \
    d $B/mpi                && \
    d $B/u/mesh/force       && \
    d $B/utils              && \
    d $B/utils/nvtx         && \
    d $B/utils/string      
