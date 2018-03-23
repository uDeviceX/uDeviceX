D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/algo/scan      && \
    d $B/clist          && \
    d $B/conf           && \
    d $B/coords         && \
    d $B/d              && \
    d $B/farray         && \
    d $B/fluforces      && \
    d $B/fluforces/bulk && \
    d $B/fluforces/halo && \
    d $B/frag           && \
    d $B/io/txt         && \
    d $B/math/rnd       && \
    d $B/mpi            && \
    d $B/pair           && \
    d $B/parray         && \
    d $B/u/bulk         && \
    d $B/utils          && \
    d $B/utils/string  
