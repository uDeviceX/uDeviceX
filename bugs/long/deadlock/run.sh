#!/bin/bash

s=../../../src
u=u/solid
u.conf $s $u base.h h/`u.host`
mkdir -p u # bug in u.dir

(u.make clean && u.make -j) > /dev/null
u.strtdir
sh runfile
