M=c

CFLAGS=-O0 -g -Wall
LDFLAGS=-lm
$M: $M.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LDFLAGS) -o $@


.PHONY: clean
clean:; rm -f $M
