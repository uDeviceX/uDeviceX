#!/bin/bash

c=1.8
f=~/googlex/

./sdf.smooth cubic $c $f  o.dat

sdf.2bov o.dat o.bov
