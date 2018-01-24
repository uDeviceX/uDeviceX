D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d        && \
    d $B/io/punto && \
    d $B/mpi      && \
    d $B/u/bulk   && \
    d $B/utils   
