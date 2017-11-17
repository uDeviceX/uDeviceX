D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d       && \
    d $B/frag    && \
    d $B/hforces && \
    d $B/mpi     && \
    d $B/rnd     && \
    d $B/u/map   && \
    d $B/utils  
