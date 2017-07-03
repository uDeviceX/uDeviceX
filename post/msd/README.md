# Diffusion

A tool for measuring MSD (Mean Squared Displacement).

## Installation

```sh
make install
```
requires `bop` repository (https://gitlab.ethz.ch/mavt-cse/bop)

## usage:

```sh
po.msd <X> <Y> <Z> rr-*.bop -- ii-*.bop
```
-`<X> <Y> <Z>` : dimensions of the domain  
-`rr-*.bop` files containing positions of particles  
-`ii-*.bop` files containing global ids of particles  

Assumes rr and ii files are in the same order.
