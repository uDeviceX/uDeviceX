# rbc

* [com](com) compute center of mass
* [main](main) initialization, restart

* [force](force) internal forces
* [force/area_volume](force/area_volume) compute area and volume of
  the mesh

* [adj](adj) adjacency list --- manage a structure to pack mesh data
* [gen](gen) generate `pp` from cell template (`rbc.off`) and initial
  condition files (`rbcs-ic.txt`)

* [rnd](rnd)         random numbers for internal forces
* [rnd/api](rnd/api) low level api for random number generator

See also [u/rbc](u/rbc), [src/test/rbccom](src/test/rbccom),
[src/test/rbc](src/test/rbc)
