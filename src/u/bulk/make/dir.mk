D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan       && \
    d $B/clist           && \
    d $B/conf            && \
    d $B/coords          && \
    d $B/d               && \
    d $B/fluforces       && \
    d $B/fluforces/bulk  && \
    d $B/fluforces/halo  && \
    d $B/frag            && \
    d $B/io/txt          && \
    d $B/math/rnd        && \
    d $B/mpi             && \
    d $B/pair            && \
    d $B/struct/farray   && \
    d $B/struct/parray   && \
    d $B/struct/pfarrays && \
    d $B/u/bulk          && \
    d $B/utils           && \
    d $B/utils/nvtx      && \
    d $B/utils/string   
