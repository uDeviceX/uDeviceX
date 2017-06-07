#!/bin/bash

d=panda:/scratch/googleX
r=vessels_small_mirrored # remote and local name
l=small                  #

rsync -avz $d/$r.sdf $l.dat # (sic)
rsync -avz $d/$r.h5  $l.h5
rsync -avz $d/$r.xmf $l.xmf

sdf.shufle yxz small.dat small.rot.dat # shufle x and y
