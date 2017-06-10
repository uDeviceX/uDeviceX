BIN = $(HOME)/bin # where to install
p=sdf
# what to install
PROGS = octave-qf $p.2bov $p.2per $p.2txt $p.cut $p.filter \
	$p.reflect $p.shuffle $p.smooth
install:; mkdir -p $(BIN) && cp $(PROGS) $(BIN)
.PHONY: install
