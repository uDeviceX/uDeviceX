#!/bin/bash

# a script to build cgal and all its dependencies using pkgsrc [1]
# Usage:
#   customize src pkg_prefix and cgal_prefix
#   and run
# [1] https://www.pkgsrc.org

set -eu
src=$SCRATCH
pkg_prefix=$SCRATCH/prefix/pkgsrc
cgal_prefix=$SCRATCH/prefix/cgal

export PATH=/usr/bin:$PATH # force default tools (gcc, sed, git, ...)

cgal_depends() {
    cd "$src" &&
	rm -rf pkgsrc &&
	git clone --depth 1  https://github.com/jsonn/pkgsrc.git &&
	rm -rf -- "${pkg_prefix?error}" &&
	cd pkgsrc/bootstrap &&
	printf "MAKE_JOBS=\t24\nSKIP_LICENSE_CHECK=\tyes\n" > mk.conf &&
	./bootstrap --prefix="$pkg_prefix" --ignore-user-check --make-jobs 24 --mk-fragment mk.conf &&
	cd ../math/cgal &&
	$pkg_prefix/bin/bmake depends
}

cgal() {
    cd "$src" &&
	VER=4.9 &&
	wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-${VER}/CGAL-${VER}.tar.xz &&
	tar xf CGAL-${VER}.tar.xz &&
	mkdir -p CGAL-BUILD &&
	cd CGAL-BUILD &&
	export PATH="$pkg_prefix"/bin:"$PATH" &&
	cmake ../CGAL-4.9 -DCMAKE_INSTALL_PREFIX:PATH="$cgal_prefix" &&
	make && make install
}

r_printf() { # [r]eadable printf
    printf "$@" | awk '/^[ \t]*$/{next} {sub("^[ \t]*", ""); print}'
}

info() {
    r_printf "
      pkg_prefix=$pkg_prefix
      cgal_prefix=$cgal_prefix
      export PATH=\"\$cgal_prefix/bin:\$pkg_prefix/bin:\$PATH\"
    "
}

cgal_depends
cgal
info
