# input:
# M: a list of target to install, main and main0 are installed a $D
# and ${D}0, other files are isntalled as ${filename}
# D: directory name
# MPATH: where to install

install: $M
	@mkdir -p $(MPATH)
	@for f in $M; \
         do case "$$f" in \
              main)  n="$D"     ;; \
              main0) n="${D}0"  ;; \
              *)     n="$$f"    ;; \
            esac; \
	    t="$(MPATH)/$P.$$n"; \
	    cp "$$f" "$$t"; \
	    echo "install '$$t'"; \
	 done

.PHONY: install
