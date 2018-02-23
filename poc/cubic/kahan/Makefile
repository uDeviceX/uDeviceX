M=c

CFLAGS=-O0 -g -Wall -Wextra
LDFLAGS=-lm
$M: $M.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LDFLAGS) -o $@


.PHONY: clean
clean:; rm -f $M
