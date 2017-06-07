RUNDIR=`pwd`
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

cd ${SRCDIR}

export PATH=../tools:$PATH

XS=8
YS=8
ZS=4

G=0.05

argp .conf.test.h                                                       \
     -tend=5.0 -steps_per_dump=100 -walls -wall_creation_stepid=0000    \
     -part_dumps                                                    \
     -gamma_dot=$G -rbcs -rcyl=1.5 -pin_com=true -dt=1e-3 -shear_y      \
     -rbc_mass=1.f -XS=${XS} -YS=${YS} -ZS=${ZS} -kBT=1e-6              \
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
