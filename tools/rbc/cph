#!/usr/bin/awk -f
# generate coordiantes of Hexagonal Close Packing of spehres

function abs(x)  { return x<0?x*-1:x}
function asin(x) { return atan2(x, sqrt(abs(1-x*x))) }
function pi() { return 2*asin(1)}

function req_var(v, n) {
    if (length(v)!=0) return
    printf "(cell-placement-hcp.awk) `%s' should be given as a parameter (-v %s=<value>)\n",
	n, n
    exit 2
}

function indomain(x, y, z) {
    if (x+r>Lx) return 0
    if (x-r<0 ) return 0

    if (y+r>Ly) return 0
    if (y-r<0 ) return 0

    if (z+r>Lz) return 0
    if (z-r<0 ) return 0

    return 1
}

BEGIN {
    req_var(A, "A"); req_var(sc, "sc"); req_var(reff, "reff");
    req_var(Lx, "Lx"); req_var(Lx, "Ly"); req_var(Lx, "Lz");

    # takes surface area `A' and two correction coefficient `reff'
    # and `sc'
    r  = sqrt(A)/(2*sqrt(pi()))
    r *= sc
    r *= reff

    nmax = 100  # should be a big number, TODO:
    for (i=0; i<nmax; i++) {
	for (j=0; j<nmax; j++) {
	    for (k=0; k<nmax; k++) {
		# https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres#Simple_hcp_lattice
		x = 2*i + ((j+k) % 2)
		y = sqrt(3) * (j + 1/3*(k % 2))
		z = 2*sqrt(6)/3*k
		x*=r; y*=r; z*=r

		if (indomain(x, y, z)) {
		    print x, y, z
		    ngen++
		}
	    }
	}
    }
    printf "(cell-placement-hcp.awk) generated: %d cells\n", ngen > "/dev/stderr"
}
