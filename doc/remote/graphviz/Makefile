all: deps.png

%.png: %.gv
	dot -Tpng $< -o $@ 
