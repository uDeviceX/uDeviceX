#!/bin/bash

echo run | u.conf ../../src conf.base.h
(u.make clean && u.make -j) > /dev/null
u.strtdir
sh runfile
