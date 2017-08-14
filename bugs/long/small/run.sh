#!/bin/bash

u.conf ../../../src  base.h h/`u.host`

(u.make clean && u.make -j) > /dev/null
u.strtdir
sh runfile

# nvprof ./udx 2>e
# grep set    e
