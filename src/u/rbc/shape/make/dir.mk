D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d           && \
    d $B/io/off      && \
    d $B/math/linal  && \
    d $B/math/rnd    && \
    d $B/math/tform  && \
    d $B/mpi         && \
    d $B/parser      && \
    d $B/rbc/adj     && \
    d $B/rbc/adj/edg && \
    d $B/rbc/shape   && \
    d $B/u/rbc/shape && \
    d $B/utils      
