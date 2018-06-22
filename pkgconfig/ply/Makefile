S = ../..

include $S/conf.mk
TRG = ply
P = $(TRG).pc
SED = sed

install: $P; u.install $P $(PKG)
$P: main.pc.in
	$(SED) "s,@TRG@,$(TRG),g; s,@INC@,$(INC),g; s,@LIB@,$(LIB),g" $< > $@

clean:; rm -f $P

.PHONY: clean install
