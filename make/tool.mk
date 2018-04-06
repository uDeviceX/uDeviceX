# input:
# P: installation prefix (like u)
# M: a list of target to install, main and main0 are replaced by $D
#    and ${D}0
# D: directory name
# B: where to install

install: $M
	@mkdir -p $B
	@for f in $M; \
         do case "$$f" in \
              main)  n="$D"     ;; \
              main0) n="${D}0"  ;; \
              *)     n="$D.$$f" ;; \
            esac; \
	    t="$B/$P.$$n"; \
	    cp "$$f" "$$t"; \
	    echo "install '$$t'"; \
	 done

.PHONY: install
