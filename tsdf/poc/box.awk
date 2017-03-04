function abs(x) {
    return x > 0 ? x : -x
}

function di(d, dlo, dhi)  {
    return \
	d < dlo ? dlo - d : \
	d > dhi ? d - dhi : \
	0
}

function de(d, dlo, dhi,    dc) {
    dc = dlo + 0.5*(dhi - dlo)
    return d > dc ? abs(d - dhi) : abs(d - dlo)
}

function min(a, b) {
    return a < b ? a : b
}

function min3(a, b, c) {
    return min(a, min(b, c))
}

function sq(x) {
    return x*x
}

function in_interval(d, dlo, dhi) {
    return \
	d < dlo ? 0 : \
	d > dhi ? 0 : \
	1
}

function in_box(x, y, z, xlo, xhi, ylo, yhi, zlo, zhi) {
    return \
	in_interval(x, xlo, xhi) && \
	in_interval(y, ylo, yhi) && \
	in_interval(z, zlo, zhi)
}

{
    xlo = ylo = zlo = 0
    xhi = yhi = zhi = 1

    x = $1; y = $2; z = $3

    dX2 = sq(de(x, xlo, xhi)) + sq(di(y, ylo, yhi)) + sq(di(z, zlo, zhi))
    dY2 = sq(di(x, xlo, xhi)) + sq(de(y, ylo, yhi)) + sq(di(z, zlo, zhi))
    dZ2 = sq(di(x, xlo, xhi)) + sq(di(y, ylo, yhi)) + sq(de(z, zlo, zhi))

    dR2 = min3(dX2, dY2, dZ2)
    dR  = sqrt(dR2)
    print in_box(x, y, z, xlo, xhi, ylo, yhi, zlo, zhi) ? dR : -dR
}
