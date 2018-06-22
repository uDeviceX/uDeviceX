CFLAGS = -O2 -g

test: test.o ply.o

%.c.o: $(CC) $(CFLAGS) $< -c -o $@

test.o: ply.h
ply.o: ply.h

.PHONY: clean

clean:; rm -f test ply.o test.o
