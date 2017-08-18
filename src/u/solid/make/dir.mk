D = @d () { test -d "$$1" || mkdir "$$1"; } && \
    d         $B && \
    d   $B/clist && \
    d       $B/d && \
    d     $B/dpd && \
    d    $B/dpdr && \
    d    $B/dual && \
    d     $B/flu && \
    d $B/hforces && \
    d      $B/io && \
    d       $B/l && \
    d $B/mbounce && \
    d   $B/mcomm && \
    d   $B/mdstr && \
    d    $B/mesh && \
    d   $B/odstr && \
    d   $B/rdstr && \
    d     $B/rig && \
    d     $B/rnd && \
    d    $B/scan && \
    d     $B/sdf && \
    d  $B/tcells && \
    d $B/u/solid && \
    d    $B/wall
