RUNDIR=`pwd`
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

# Note: before running this, switch off solid bounce back!

cd ${SRCDIR}

export PATH=../tools:$PATH

XS=16
YS=32
ZS=16

argp .conf.test.h                                                       \
     -tend=100.0 -steps_per_dump=1000 -walls -wall_creation_stepid=5000 \
     -field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000          \
     -pushflow -hydrostatic_a=0.05 -dt=1e-3                          \
     -rbcs -rsph=5 -rbc_mass=1.f  -pin_com=false                        \
     -XS=${XS} -YS=${YS} -ZS=${ZS}                                      \
     > .conf.h

make clean && make -j && make -C ../tools

cp udx ${RUNDIR}

cd ${RUNDIR}

rm -rf h5 *.txt sdf.dat

echo "extent ${XS} ${YS} ${ZS}
N          32
obj_margin 3.0

# normal goes from inside wall to outside
plane point xc 0.9*Ly zc normal 0 -1 0
plane point xc 0.1*Ly zc normal 0  1 0" > yplates.tsdf

tsdf yplates.tsdf sdf.dat

cat run.sh > run.back.sh

./udx
