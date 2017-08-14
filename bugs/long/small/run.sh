#!/bin/bash

u.conf ../../../src conf.base.h <<!
run
!

(u.make clean && u.make -j) > /dev/null
u.strtdir

nvprof ./udx 2>e
grep memset    e
