#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "usage: $0 dirname"
  exit 1
fi

# need ri (git@gitlab.ethz.ch:mavt-cse/ri.git)

r $1 ^ cd sdf/gx/ '&&' sh copy.sh
r $1 ^ cp .cache.Makefile.amlucas.daint .cache.Makefile '&&' sh c.sh
