#!/bin/bash

set -eu

# load utilities
. u/`u.host`

n=1
d=/scratch/snx3000/lisergey/u
Time=00:15:00
u=u/safe
s=../src
Restart=1

(cd $s/../cmd; make)
u.conf $s $u base.h <<EOF
       DBG_PEEK
       RESTART=true
       ODSTR_SAFE
       run
EOF
u.make -j > make.log
safe_cp "$d/$n/sdf.dat" .

x=$n y=1 z=1

if test $Restart -eq 1
then u.re.pick
else u.strtdir  . $x $y $z
fi

u.batch   $x $y $z ./udx $Time
