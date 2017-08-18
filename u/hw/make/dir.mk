D = @d () { test -d "$$1" || mkdir "$$1"; } && \
    d          $B && \
    d $B/../u/hw/ && \
    d        $B/l
