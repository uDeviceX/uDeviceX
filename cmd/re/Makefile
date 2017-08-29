BIN = $(HOME)/bin
B = $(BIN)

# files and prefixes (with dot)
x = u.re.
X = push list clear conf pick

y = u.re.pick
Y = Pick/0 Pick/1 Pick/2 Pick/3

install:
	@echo install re tools to $B
	@mkdir -p -- $B
	for f in $X; do cp -- $$f $B/$x`basename $$f`; done
	for f in $Y; do cp -- $$f $B/$y`basename $$f`; done

.PHONY: install
