subdirs=build_smesh disp drop field msd
.PHONY: install $(subdirs)

install: $(subdirs)

$(subdirs):
	make install -C $@

clean:
	make clean -C build_smesh
	make clean -C disp
	make clean -C msd

.PHONY: $(subdirs) clean install
