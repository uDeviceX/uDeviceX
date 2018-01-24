D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords && \
    d $B/d      && \
    d $B/dbg    && \
    d $B/io/txt && \
    d $B/mpi    && \
    d $B/parser && \
    d $B/u/dbg  && \
    d $B/utils 
