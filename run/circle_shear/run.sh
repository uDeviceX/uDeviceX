RUNDIR=`pwd`
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

cd ${SRCDIR}

export PATH=../tools:$PATH

XS=64
YS=64
ZS=8

G=0.05

# cheap trick for keeping Peclet number fixed
kBT=`awk "BEGIN{ printf \"%.6e\n\", 0.27925268 * $G }"`

argp .conf.test.h                                                       \
     -tend=300.0 -steps_per_dump=1000 -walls -wall_creation_stepid=5000 \
     -field_dumps -part_dumps -steps_per_hdf5dump=1000              \
     -gamma_dot=$G -rbcs -spdir=2 -sbounce_back -dt=1e-3 -shear_y       \
     -rbc_mass=1.f -XS=${XS} -YS=${YS} -ZS=${ZS} -kBT=$kBT              \
     > .conf.h

(make clean && make -j && make -C ../tools) > /dev/null

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

echo `awk "BEGIN{ printf \"%.6e %.6e %.6e\n\", 0.5*${XS}, 0.5*${YS}, 0.5*${ZS} }"` > ic_solid.txt

GEN=${GITROOT}/pre/meshgen/gencylinder

${GEN} mesh_solid.ply 5 48 ${ZS} 32

cat run.sh > run.back.sh

./udx
