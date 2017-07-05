# disp

a postprocessing tool for computing displacement of particles.
Displacement is defined as difference of positions between 2 time samples, `dr(t) = r(t+1) - r(t)`.

## Installation

```sh
make install
```
requires `bop` repository (https://gitlab.ethz.ch/mavt-cse/bop)

## Usage:

```sh
po.disp <X> <Y> <Z> rr-*.bop -- ii-*.bop
```
-`<X> <Y> <Z>` : dimensions of the domain  
-`rr-*.bop` files containing positions of particles  
-`ii-*.bop` files containing global ids of particles  

Assumes rr and ii files are in the same order.
