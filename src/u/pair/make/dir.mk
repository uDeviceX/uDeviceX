D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d      && \
    d $B/mpi    && \
    d $B/pair   && \
    d $B/parser && \
    d $B/u/pair && \
    d $B/utils 
