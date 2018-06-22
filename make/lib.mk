# input:
# TRG: target base name
# O: object files
# I: include files (headers)
# CFLAGS, U_CFLAGS, CC:
# LIB: where to install lib$(TRG).a
# INC: where to install $I
#
# requres u.install

L = lib$(TRG).a

$L: $O; ar rv $@ $H && ranlib $@
%.o: %.c; $(CC) $(CFLAGS) $(U_CFLAGS) -c -o $@ $<

install: $L $H
	u.install $L $(LIB)
	u.install $I $(INC)

clean:; rm -f $O $L

.PHONY: clean install
