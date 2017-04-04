RUNDIR=`pwd`
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

cd ${SRCDIR}

export PATH=../tools:$PATH

XS=64
YS=32
ZS=8

argp .conf.test.h                                                       \
     -tend=300.0 -steps_per_dump=1000 -walls -wall_creation_stepid=5000 \
     -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000          \
     -gamma_dot=0.05 -rbcs -rcyl=5 -pin_com=true -dt=1e-3 -shear_y      \
     -rbc_mass=1.f -XS=${XS} -YS=${YS} -ZS=${ZS}                        \
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
