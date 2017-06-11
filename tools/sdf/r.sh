#!/bin/bash

make

f=~/googlex/fixed.sdf # 512 512 48

process () {
    sdf.cut :$xh :$yh : $f  o.dat
    sdf.2per o.dat      $b.dat
    sdf.2bov $b.dat     $b.bov
}

xh=128 yh=$xh b=$xh
process

xh=210 yh=$xh b=$xh
process
