D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d            && \
    d $B/l            && \
    d $B/scan         && \
    d $B/u/odstr/halo
