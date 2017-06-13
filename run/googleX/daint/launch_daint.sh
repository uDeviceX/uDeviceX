# !/bin/bash

module load GSL cray-hdf5-parallel cudatoolkit

BNAME=small_n4
SRCDIR=$HOME/uDeviceX/uDX_gx


DEPLOYDIR=`pwd`

mydate=`date | awk '{print $6$2$3"_"$4}'`
RUNDIR=$SCRATCH/GOOGLEX/${BNAME}_${mydate}


cd $SCRATCH; tmpname=`mktemp -d tmp.XXXXXXXXX`
cd -
COMPILEDIR=$SCRATCH/${tmpname}


cp -r ${SRCDIR}/* ${COMPILEDIR}/
(
cd ${COMPILEDIR}/src
    cp ${DEPLOYDIR}/../conf.h conf.h
    cp ${DEPLOYDIR}/cache.Makefile.daint .cache.Makefile
    make clean
    make -j
    make -C ${SRCDIR}/tools
)


mkdir -p ${RUNDIR}
(
cd ${RUNDIR}
    cp ${COMPILEDIR}/src/udx .
    cp ${DEPLOYDIR}/run_daint.sh .
    cp ${COMPILEDIR}/src/conf.h conf.h
    . run_daint.sh
)

rm -rf ${COMPILEDIR}
