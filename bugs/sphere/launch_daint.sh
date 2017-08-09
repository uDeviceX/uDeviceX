# !/bin/bash
#
# Script for launcing uDeviceX on daint:$SCRATCH
# run as: ./launch_daint.sh
#
# Explanation of script variables:
# --------------------------------
# GITROOT:      ??/uDeviceX
# DEPLOYDIR:    ??/uDeviceX/run/couette
# COMPILEDIR:   $SCRATCH/tmp.XXXXXXXXX
# RNAME:        couette_R4_HA0.1_DT0.0001_02Aug2017
# RUNDIR:       $SCRATCH/UDEVICEX/RNAME

mkdir -p $SCRATCH/UDEVICEX

GITROOT=`git rev-parse --show-toplevel`
DEPLOYDIR=`pwd`
COMPILEDIR=`mktemp -d $SCRATCH/tmp.XXXXXXXXX`
RNAME=`awk -v n=${RNAME} 'BEGIN {if(length(n) > 0) print n; else print "test"}'`
RUNDIR=$SCRATCH/UDEVICEX/${RNAME}_`date | awk '{print $2$3"_"$4}'`

# Copy+compile in $SCRATCH
cp -r ${GITROOT}/* ${COMPILEDIR}/
mkdir -p ${COMPILEDIR}/build
cp ./* ${COMPILEDIR}/build

cd ${COMPILEDIR}/build

    # Create Makefile
    ${COMPILEDIR}/src/configure ${COMPILEDIR}/src

    # Compile utilities
    #make -C ${GITROOT}/tools/rbc install        > /dev/null
    #make -C ${GITROOT}/tools install            > /dev/null
    #make -C ${GITROOT}/post/build_smesh install > /dev/null
    (cd ${GITROOT}/cmd; make ;                   > /dev/null)

    # Edit configuration script
    u.conf ${COMPILEDIR}/src conf.base.h <<EOF

    run
EOF

    # Compile
    { make clean && u.make -j ; } > /dev/null
cd -

# Copy to $RUNDIR
mkdir -p ${RUNDIR}
cp ${COMPILEDIR}/build/udx ${RUNDIR}/
cp ${COMPILEDIR}/build/launch_daint.sh ${RUNDIR}/
cp ${COMPILEDIR}/build/run_daint.sh ${RUNDIR}/
cp ${COMPILEDIR}/build/conf.h ${RUNDIR}/

(
cd $RUNDIR
. run_daint.sh
)


# Rerun with restarts
cd ${COMPILEDIR}/build
    echo -e "#define RESTART true" >> ${COMPILEDIR}/build/conf.h
    { make clean && u.make -j ; } > /dev/null
cd -
# Copy to $RUNDIR
cp ${COMPILEDIR}/build/udx ${RUNDIR}/
cp ${COMPILEDIR}/build/launch_daint.sh ${RUNDIR}/
cp ${COMPILEDIR}/build/run_daint.sh ${RUNDIR}/
cp ${COMPILEDIR}/build/conf.h ${RUNDIR}/

(
cd $RUNDIR
. run_daint.sh
)

#rm -rf ${COMPILEDIR}

