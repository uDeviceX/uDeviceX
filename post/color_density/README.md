# color density

average color density (bop format) and project it on a field (bov format)

# requirements
The two libraries [bov](https://gitlab.ethz.ch/mavt-cse/bov) and [bop](https://gitlab.ethz.ch/mavt-cse/bop).

## color.minmax

A small oversimplified tool for Hele Shaw measurements of a slice of color density. Assumes nz = 1 and one component.
prints the minimum x coordinate such as  `rho(x,y) > T0` and the maximum x coodinate such as `rho(x,y) < T1`.
