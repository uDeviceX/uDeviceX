#!/bin/bash

make

f=/home/amlucas/packgx/solid/src/sdf/gx/small.rot.dat # 100 80 24
# f=~/googlex/vessels.sdf # 512 512 48

process () {
    sdf.cut :$xh :$yh : $f  o.dat
    sdf.2per o.dat      $b.dat
    sdf.2bov $b.dat     $b.bov
}

xh=40 yh=40 b=tiny
process
