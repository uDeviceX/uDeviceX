#!/bin/bash

#GDOT=0.0500
#BB=nobounce
GDOT=$1
BB=$2

export CRAY_CUDA_MPS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT=`pwd`/$0
RUNDIR=/scratch/snx3000/amlucas/${BB}/gdot-${GDOT}
GITROOT=`git rev-parse --show-toplevel`
SRCDIR=${GITROOT}/src

# setup and compile

cd ${SRCDIR}

XS=64
YS=64
ZS=8

argp .conf.test.h                                                        \
     -tend=1000.0 -steps_per_dump=1000 -walls -wall_creation_stepid=5000 \
     -hdf5part_dumps -steps_per_hdf5dump=1000                            \
     -gamma_dot=${GDOT} -rbcs -rcyl=5 -pin_com=true -dt=1e-3 -shear_y    \
     -rbc_mass=1.f -XS=${XS} -YS=${YS} -ZS=${ZS}                         \
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
#SBATCH --time=03:00:00
#SBATCH --partition=normal

srun -C gpu -n ${nnodes} --ntasks-per-node=${ntasks} -c ${ncpus} ./udx
EOF
