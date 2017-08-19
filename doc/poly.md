# conditional compilation (polymorphous)

## Conditional object file

Controlled by passing `-I$S/fun/cpu0` and `-I$S/fun` to files inside
`fun/`

	fun/int.h
	   /cpu0/imp.cu
			 special.h
	   /cuda/imp.cu
			 special0.h
			 special1.h
	   common.h

`cpu0/imp.cu` contains

	#include "common.h"
	#include "special.h"

## Conditional header

Only global: one directory `g/` for entire project.  Controlled by
passing `-I$S/g/cpu0` to all files.

	g/cpu0/api.h
	 /cuda/api.h
