#!/bin/bash

make > /dev/null

f0=$HOME/googlex/big.sdf
s0=$HOME/googlex/small.sdf
c=$1 # kernel size

smooth0 () { cp       w.dat             $b.sdf;  }
smooth1 () { sdf.smooth cubic $c w.dat  $b.sdf; }
smooth () { if test "$c" = 0; then smooth0; else smooth1; fi; }
per0() { : ; }

per1() {
    sdf.2per  w.dat             w0.dat
    mv       w0.dat              w.dat
}

per() { if test "$b" = small; then per0; else per1; fi; }

process () {
    sdf.cut :$xh :$yh : $f     w.dat

    per
    smooth

    sdf.2bov $b.sdf            $b.bov
}

p () {
    printf '%03d' "$1"
}

d=~/googlex/$c
mkdir -p $d/bov

cd $d

f=$f0 xh=25  yh=$xh b=`p $xh`; process
f=$f0 xh=50  yh=$xh b=`p $xh`; process
f=$f0 xh=100 yh=$xh b=`p $xh`; process
f=$f0 xh=150 yh=$xh b=`p $xh`; process
f=$f0 xh=200 yh=$xh b=`p $xh`; process
f=$s0 xh=    yh=    b=small;   process
f=$f0 xh=    yh=    b=big;     process

mv *.bov *.values bov/
rm -f w0.dat w.dat

#######
# echo 0 1 2 4 8 16 32 64 | xargs -n 1 sh r.sh
# rsync -rav $HOME/googlex/ falcon:/tmp/googlex
