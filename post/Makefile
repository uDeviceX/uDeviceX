subdirs=build_smesh disp drop field msd binning stress
.PHONY: install $(subdirs)

install: $(subdirs)

$(subdirs):
	make install -C $@

clean:
	make clean -C build_smesh
	make clean -C disp
	make clean -C msd
	make clean -C color_density
	make clean -C binning

test:
	make test -C color_density
	make test -C disp
	make test -C binning


.PHONY: $(subdirs) clean install test
