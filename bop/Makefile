M=make
include $M/usr.mk
include $M/common.mk

conf=bop-config

all: converters utils tools libbop install installtools tests installconfig

install: libbop ; (cd src; make install)

installtools: converters tools
	(cd converters; make install)	
	(cd tools;      make install)

libbop: ; (cd src; make)

utils:      libbop install ;  (cd utils;      make)
converters: libbop install ;  (cd converters; make)
tools:      libbop install ;  (cd tools;      make)
tests:      libbop install ;  (cd test;       make)

test: all
	(cd converters; make test)
	(cd test;       make test)
	(cd tools;      make test)
	(cd utils;      make test)

clean:
	(cd converters; make clean)
	(cd src;        make clean)
	(cd test;       make clean)
	(cd utils;      make clean)	
	(cd tools;      make clean)
	rm -rf $(conf)

edit = sed \
	-e 's|@prefix[@]|$(prefix)|g'

$(conf): $(conf).in
	@$(edit) $@.in > $@.tmp
	@chmod +x $@.tmp
	@mv $@.tmp $@
	@echo created $@

installconfig: $(conf)
	@mkdir -p $(INST_BIN)
	@cp $(conf) $(INST_BIN)

.PHONY: clean install test tests libbop utils converters tools installtools installconfig
