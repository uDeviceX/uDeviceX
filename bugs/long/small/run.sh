#!/bin/bash

s=../../../src
u.conf $s base.h h/`u.host`
(u.make clean && u.make -j) > /dev/null
u.strtdir
sh runfile
