D = cmd

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D; do install0; done

clean:; cd src; make clean
test:; atest test/*

.PHONY: clean test
