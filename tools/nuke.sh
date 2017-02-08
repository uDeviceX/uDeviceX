#!/bin/bash

# nuke files in uD: Try to compile if it compiles without th file nuke
# it

match () { # return 1 if matching
    echo "$f" | awk '!/'"$@"'/{exit 1}'
}

compile () { # compile uD silently
    (
	git clean -f -d -x > /dev/null
	cd geom-wrapper
	(. ~/.cgal.bashrc ; cmake . ) 1>/dev/null
	make -j                       1>/dev/null
	cd ../mpi-dpd
	cp .cache.Makefile.lisergey.falcon .cache.Makefile
	make -j                        >/dev/null 2>/dev/null
    )
}

cd hack.rbc_shear

for f in `find . -name '*.cu' -o -name '*.h' -o -name '*.cpp'`
do
    if match geom-wrapper; then continue; fi # skip a directory
    if match device-gen  ; then continue; fi

    rm -rf -- "${f?error}"
    if compile
    then
	printf "bad : %s\n" "$f"
    else
	printf "good: %s\n" "$f"
	# restore file
	git checkout "$f"
    fi
done
