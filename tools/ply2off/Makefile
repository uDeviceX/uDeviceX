S = ../..
include $S/conf.mk

M = main
P = u
B = $(BIN)
D = ply2off
include $S/make/tool.mk

U_CFLGAS = -Wall -Wextra `u.pkg-config --cflags ply`
U_LDFLAGS = `u.pkg-config --libs ply`

%.o: %.c;   $(CC) $(CFLAGS) $(U_CFLAGS) -c -o $@ $<
$M: $M.o;   $(CC) $<  $(LDFLAGS) $(U_LDFLAGS) -o $@
$M.o: $M.c; $(CC) $(CFLGAS) $(U_CFLGAS) $< -c -o $@

clean:; rm -f $M.o $M
