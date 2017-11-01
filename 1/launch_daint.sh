# !/bin/bash
#
# Script for launcing uDeviceX on daint:$SCRATCH
# run as: ./launch_daint.sh <Level> <RNAME> <Dp> <Yp>
#
# --------------------------------
# Explanation of script variables:
# --------------------------------
# GITROOT:      ??/uDeviceX
# DEPLOYDIR:    ??/uDeviceX/run/couette
# COMPILEDIR:   $SCRATCH/tmp.XXXXXXXXX
# RNAME:        couette_R4_HA0.1_DT0.0001_02Aug2017
# RUNDIR:       $SCRATCH/UDEVICEX/RNAME
#
#
# --------------------------------
# Parameter parsing:
# --------------------------------
L=$1 # defined here and used later in run_daint.sh
RNAME=$2
D=$3
iyp=$4

# --------------------------------
mkdir -p $SCRATCH/UDEVICEX

GITROOT=`cd ../../; git rev-parse --show-toplevel` #assumes we are inside the udx src code repo
DEPLOYDIR=`pwd`
COMPILEDIR=`mktemp -d $SCRATCH/tmp.XXXXXXXXX`
RNAME=`awk -v n=${RNAME} 'BEGIN {if(length(n) > 0) print n; else print "test_solid"}'`
RUNDIR=$SCRATCH/UDEVICEX/${RNAME}_`date | awk '{print $2$3"_"$4}'`
GIT_COMMIT=`./git_branch.sh`

# Copy+compile in $SCRATCH
cp -r ${GITROOT}/* ${COMPILEDIR}/
mkdir -p ${COMPILEDIR}/build
cp ./* ${COMPILEDIR}/build

cd ${COMPILEDIR}/build
    # Compile utilities
    (cd ../cmd; make ;                   > /dev/null)

    # Edit configuration script
    u="u/x"
    s=../src
    u.conf $s $u conf.base.h <<EOF
    run
EOF

    # Compile
    u.make clean -s
    u.make -j -s

    # Copy to $RUNDIR
    mkdir -p ${RUNDIR}
    cp ${COMPILEDIR}/build/udx ${RUNDIR}/
    cp ${COMPILEDIR}/build/launch_daint.sh ${RUNDIR}/
    cp ${COMPILEDIR}/build/run_daint.sh ${RUNDIR}/
    cp ${COMPILEDIR}/build/conf.h ${RUNDIR}/

cd $RUNDIR
    source ./run_daint.sh

cd ${DEPLOYDIR}
    echo ${GIT_COMMIT} > $RUNDIR/git_commit.txt

rm -rf ${COMPILEDIR}
