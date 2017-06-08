#!/bin/bash

d=panda.ethz.ch:/scratch/googleX

r=vessels_small_mirrored # remote and local name
l=small                  #
rsync -avz $d/$r.sdf $l.dat # (sic)
rsync -avz $d/$r.h5  $l.h5
rsync -avz $d/$r.xmf $l.xmf

r=small.rot
l=small.rot
rsync -avz $d/$r.dat $l.dat
