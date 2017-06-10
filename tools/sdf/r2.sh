#!/bin/bash

c=10.0
#f=~/googlex/small.rot.dat
f=~/googlex/128.dat
#f=~/googlex/256.dat

./sdf.smooth cubic $c $f  o.dat

sdf.2bov o.dat o.bov
