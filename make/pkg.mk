# input:
# TRG, INC, LIB
# P: pc file

SED = sed

install: $P; u.install $P $(PKG)
$P: main.pc.in
	$(SED) "s,@TRG@,$(TRG),g; s,@INC@,$(INC),g; s,@LIB@,$(LIB),g" $< > $@

clean:; rm -f $P

.PHONY: clean install
