D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/clist      && \
    d $B/cnt        && \
    d $B/d          && \
    d $B/dpd        && \
    d $B/dpdr       && \
    d $B/dual       && \
    d $B/flu        && \
    d $B/fsi        && \
    d $B/hforces    && \
    d $B/io         && \
    d $B/l          && \
    d $B/mbounce    && \
    d $B/mcomm      && \
    d $B/mdstr      && \
    d $B/mesh       && \
    d $B/odstr      && \
    d $B/odstr/halo && \
    d $B/odstr/pack && \
    d $B/rbc        && \
    d $B/rdstr      && \
    d $B/rig        && \
    d $B/rnd        && \
    d $B/scan       && \
    d $B/sdf        && \
    d $B/sdstr      && \
    d $B/tcells     && \
    d $B/wall      
