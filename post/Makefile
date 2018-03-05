subdirs=disp drop field msd binning stress
.PHONY: install $(subdirs)

install: $(subdirs)

$(subdirs):
	(cd $@; make install)

clean:
	(cd binning; make clean)
	(cd disp;    make clean)
	(cd msd;     make clean)


test:
	(cd binning; make test)
	(cd disp;    make test)

.PHONY: $(subdirs) clean install test
