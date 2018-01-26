D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/coords       && \
    d $B/d            && \
    d $B/io/field     && \
    d $B/io/field/h5  && \
    d $B/io/field/xmf && \
    d $B/mpi          && \
    d $B/u/io/h5      && \
    d $B/utils       
