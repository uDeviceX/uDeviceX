# configure, compile and run

    u.conf [src] default.h <<< \
	  part_freq=10
	  run 1000
	  freeze
      part_freq=2
	  pushflow
	  run 1000

It generates

* `makefile`
* `runfile`
* `bin.1` `bin.2` directories, in `bin.[12]` it puts `conf.h` files.

command `u.make` builds `bin.[12]/udx` using `makefile`. Shell script
`runfile` runs `bin.[12]/udx`.

`runfile` runs from current directory.
