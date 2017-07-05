# MSD

A tool for measuring MSD (Mean Squared Displacement).

## Installation

```sh
make install
```
requires `bop` repository (https://gitlab.ethz.ch/mavt-cse/bop)

## usage:

```sh
po.msd <optional options> <X> <Y> <Z> rr-*.bop -- ii-*.bop
```
-`<X> <Y> <Z>` : dimensions of the domain  
-`rr-*.bop` files containing positions of particles  
-`ii-*.bop` files containing global ids of particles  

Assumes rr and ii files are in the same order.

#### Optional options:

`-t0step <t0step>` to specify the steps for starting points `t0` used for the average. (default: `N`)  
Note that a small step leads to `O(N^2)` file reads, where `N` is the number of files!
