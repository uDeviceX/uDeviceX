D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/conf   && \
    d $B/d      && \
    d $B/mpi    && \
    d $B/pair   && \
    d $B/u/pair && \
    d $B/utils 
