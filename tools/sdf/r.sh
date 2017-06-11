#!/bin/bash

make

f=~/googlex/fixed.sdf # 512 512 48
c=12.0

process () {
    sdf.cut :$xh :$yh : $f     w.dat
    sdf.2per w.dat             w0.dat
    cp       w0.dat            w.dat
    sdf.smooth cubic $c w.dat  $b.dat

    sdf.2bov $b.dat     $b.bov
}

xh=50  yh=$xh b=$xh; process
xh=100 yh=$xh b=$xh; process
xh=150 yh=$xh b=$xh; process
xh=200 yh=$xh b=$xh; process
xh=250 yh=$xh b=$xh; process
