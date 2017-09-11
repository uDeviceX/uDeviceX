#!/bin/bash

. ../generic.sh
. ./pre.sh

XS=24
YS=24
ZS=24

NX=$1; shift
NY=$1; shift
NZ=$1; shift

clean
setup
geom
echo "Domain: $Domain"

u="u/x"
s="${GITROOT}/src"

u.conf $s $u conf.base.h <<EOF
    $Domain
    run
EOF

compile

u.batch $NX $NY $NZ ./udx
