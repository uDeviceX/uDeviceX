#!/bin/bash

echo run | u.conf ../../src conf.base.h
u.make -j

u.strtdir
sh runfile
