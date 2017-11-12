D = @d () { test -d "$$1" || mkdir -p -- "$$1"; } && \
    d $B/d                     && \
    d $B/io                    && \
    d $B/io/bop                && \
    d $B/io/field              && \
    d $B/io/field/h5           && \
    d $B/io/field/xmf          && \
    d $B/math                  && \
    d $B/mpi                   && \
    d $B/rbc/com               && \
    d $B/rbc/force             && \
    d $B/rbc/force/area_volume && \
    d $B/rbc/main              && \
    d $B/u/rbc                 && \
    d $B/utils                
