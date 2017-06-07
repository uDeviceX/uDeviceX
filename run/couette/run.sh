RUNDIR=`pwd`
SRCDIR=${RUNDIR}/../../src

cd ${SRCDIR}

export PATH=../tools:$PATH

argp .conf.test.h                                                      \
     -tend=200.0 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
     -field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000         \
     -_gamma_dot=0.1 -rcyl=5 -pin_com=true -dt=1e-3 -shear_y > .conf.h

make clean && make -j && make -C ../tools

cp udx ${RUNDIR}
cp sdf/yplates1/yplates.dat ${RUNDIR}/sdf.dat

cd ${RUNDIR}

rm -rf h5 diag.txt solid_diag.txt

./udx


