#!/bin/bash

#GDOT=0.0500
#BB=nobounce
GDOT=$1
BB=$2
DT=$3

# cheap trick for keeping Peclet number fixed
kBT=`awk "BEGIN{ printf \"%.6e\n\", 0.27925268 * $GDOT }"`

export CRAY_CUDA_MPS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT=`pwd`/$0
RUNDIR=/scratch/snx3000/amlucas/ellipse_cstPe/${BB}/gdot+${GDOT}_dt+${DT}
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

# setup and compile

cd ${SRCDIR}

XS=64
YS=64
ZS=8

argp .conf.test.h                                                        \
     -tend=5000.0 -steps_per_dump=1000 -walls -wall_creation_stepid=5000 \
     -field_dumps=false -part_dumps -steps_per_hdf5dump=1000     \
     -gamma_dot=${GDOT} -dt=${DT} -shear_y                               \
     -rbcs -rbc_mass=1.f -a2_ellipse=16 -b2_ellipse=4 -pin_com=true      \
     -XS=${XS} -YS=${YS} -ZS=${ZS}  -kBT=$kBT                            \
     > .conf.h

make clean && make -j && make -C ../tools


# go to scratch

mkdir -p ${RUNDIR}

cp udx ${RUNDIR}

cd ${RUNDIR}

rm -rf h5 *.txt

echo "extent ${XS} ${YS} ${ZS}
N          32
obj_margin 3.0

# normal goes from inside wall to outside
plane point xc 0.9*Ly zc normal 0 -1 0
plane point xc 0.1*Ly zc normal 0  1 0" > yplates.tsdf

tsdf yplates.tsdf sdf.dat

cat $SCRIPT > run.back.sh

# run

nnodes=1
ntasks=1
ntasksc=1
ncpus=1

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name="${BB}-${GDOT}"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=${ncpus}
#SBATCH --constraint=gpu
#SBATCH --time=13:00:00
#SBATCH --partition=normal

srun -C gpu -n ${nnodes} --ntasks-per-node=${ntasks} -c ${ncpus} ./udx
EOF
