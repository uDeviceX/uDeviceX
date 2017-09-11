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
compile ${GITROOT}/src/
