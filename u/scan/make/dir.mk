D = @d () { test -d "$$1" || mkdir "$$1"; } && \
    d           $B && \
    d $B/../u/scan && \
    d         $B/d && \
    d         $B/l && \
    d      $B/scan
