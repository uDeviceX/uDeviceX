#!/bin/sh

# need ri (git@gitlab.ethz.ch:mavt-cse/ri.git)

RUN=run.gx.disk.sh

r $1 ^ cd sdf/gx/ '&&' sh copy.sh
r $1 ^ cp .cache.Makefile.amlucas.daint .cache.Makefile '&&' sh run.gx.disk.sh
