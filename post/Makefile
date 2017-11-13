subdirs=build_smesh disp drop field msd
.PHONY: install $(subdirs)

install: $(subdirs)

$(subdirs):
	make install -C $@

clean:
	make clean -C build_smesh
	make clean -C disp
	make clean -C msd
	make clean -C color_density

test:
	make test -C color_density
	make test -C disp


.PHONY: $(subdirs) clean install test
