# Intro

Red blood cell. [src/rbc/type.h](type.h) defines `struct rbc::Quants`

# Components

-   [adj](adj) adjacency list: a structure to pack mesh to read on
    device
-   [com](com) compute center of mass
-   [edg](edg) store information for every edge (host)
-   [force](force) internal forces
-   [gen](gen) generate `pp` from cell template (`rbc.off`) and initial
    condition files (`rbcs-ic.txt`)
-   [main](com) initialization, restart
-   [rnd](rnd) random numbers for internal forces
-   [rnd/api](rnd/api) low level api for random number generator
-   [force/area\_volume](force/area_volume) compute area and volume
-   [stretch](stretch) apply a force to every vertex of every cell,
    force is set from a file `rbc.stretch`

See also [src/u/rbc](src/u/rbc), [src/test/rbccom](src/test/rbccom),
[src/test/rbc](src/test/rbc)

# Cell templates

See [src/data/cells](src/data/cells)
