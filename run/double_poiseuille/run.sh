RUNDIR=`pwd`
SRCDIR=${RUNDIR}/../../src

cd ${SRCDIR}

export PATH=../tools:$PATH

argp .conf.test.h                                                       \
     -tend=100.0 -steps_per_dump=1000 -pushflow -doublepoiseuille    \
     -field_dumps -part_dumps -steps_per_hdf5dump=1000              \
     -hydrostatic_a=0.02 -dt=1e-3 > .conf.h
     

make clean && make -j && make -C ../tools

cp udx ${RUNDIR}

cd ${RUNDIR}

rm -rf h5 diag.txt solid_diag.txt

cat run.sh > run.back.sh

./udx


