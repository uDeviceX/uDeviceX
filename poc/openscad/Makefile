STL=cylinder.stl
OFF=cylinder.off

all: ${STL} ${OFF}

%.stl: %.scad; openscad -o $@ $<
%.off: %.scad; openscad -o $@ $<

clean:; -rm ${STL} ${OFF}
.PHONY: clean
