# !/bin/bash
#
# Script for launcing uDeviceX on daint
# run as: ./run.sh <RNAME>

set -e

# --------------------------------
# Input parameters:
# --------------------------------
# ./launch.sh RNAME SH rnd sc sfree
RNAME=$1
sc=$2
sfree=$3

# Read params from txt file
nv=`awk '{print $1}' params.txt`
XS=`awk '{print $2}' params.txt`
YS=`awk '{print $3}' params.txt`
ZS=`awk '{print $4}' params.txt`
abb=`awk '{print $5}' params.txt`
gbb=`awk '{print $6}' params.txt`
arr=`awk '{print $7}' params.txt`
grr=`awk '{print $8}' params.txt`
gc=`awk '{print $9}' params.txt`
kbt=`awk '{print $10}' params.txt`
kb=`awk '{print $11}' params.txt`
ks=`awk '{print $12}' params.txt`
X0=`awk '{print $13}' params.txt`
A0=`awk '{print $14}' params.txt`
V0=`awk '{print $15}' params.txt`
sh=`awk '{print $16}' params.txt`

arb=`awk -v r=$arr -v b=$abb 'BEGIN {print (r+b)/2.}'`
grb=`awk -v r=$grr -v b=$gbb 'BEGIN {print (r+b)/2.}'`

mkdir -p $SCRATCH/UDEVICEX

H=`pwd`

# src dir
S=`u.cp.s`

# compile dir
C=`mktemp -d $SCRATCH/tmp.XXXXXXXXX`

# run dir
RNAME=`awk -v n=${RNAME} 'BEGIN {if(length(n) > 0) print n; else print "test_shear"}'`
R=$SCRATCH/UDEVICEX/Feb15_test/${RNAME}
mkdir -p $R

function gorun() {
    cd $R
}

function gocomp() {
    cd $C
}

function goback() {
    cd $H
}

function conf() {
    U=u/x
    u.conf $S $U $H/conf.base.h <<EOF
    run
EOF
}

function compile() {
    u.make clean -s
    u.make -j -s > /dev/null 2>&1
}

function pre() {
    source ./pre.sh
}

function run() {
    sbatch sbatch.sh
    #srun -u -n ${NN} ./udx ${NX} ${NY} ${NZ}
     #./udx ${NX} ${NY} ${NZ}
}

function transfer() {
    cp $C/udx .
    cp $C/conf.h .
    cp $H/conf.cfg .
    cp $H/creator.py .
    cp $H/params.txt .
    cp $H/pre.sh .
}

function rmcomp() {
    rm -rf $C
}


gocomp
conf
compile

gorun
transfer
pre
run
goback

rmcomp
