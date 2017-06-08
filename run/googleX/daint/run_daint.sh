# !/bin/bash

(
cd ${RUNDIR}

# number of ranks
if [[ -n "${NX}" ]]; then NX=${NX}; else NX=1; fi
if [[ -n "${NY}" ]]; then NY=${NY}; else NY=1; fi
if [[ -n "${NZ}" ]]; then NZ=${NZ}; else NZ=1; fi
NN=$((${NX}*${NY}*${NZ}))

# copy walls
cp ${SRCDIR}/src/sdf/gx/vessels_mirrored.dat sdf.dat
#cp ${SRCDIR}/src/sdf/gx/vessels_small_mirrored.dat sdf.dat

# solid mesh + position
# /users/eceva/uDeviceX/uDX_gx/src/data/sphere.ply

# doesnt run w/o this
cp ${SRCDIR}/src/rbc.off .

# create sbatch script
 echo "#!/bin/bash -l
#
#SBATCH --job-name=gx
#SBATCH --time=00:30:00
#SBATCH --ntasks=${NN}
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out.%j.o
#SBATCH --constraint=gpu
#SBATCH --error=slurm_out.%j.e
#SBATCH --account=ch7
#SBATCH --partition=low

# ======START=====
module load daint-gpu
module load slurm
export CRAY_CUDA_MPS=1

srun -u -n ${NN} ./udx ${NX} ${NY} ${NZ}
# =====END====" > sbatch.sh
 
# submit on daint
sbatch sbatch.sh

)
