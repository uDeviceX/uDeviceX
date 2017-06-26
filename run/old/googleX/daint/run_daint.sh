# !/bin/bash

(
cd ${RUNDIR}
pwd

#make -C ${SRCDIR}/tools/rbc install
#make -C ${SRCDIR}/tools install
#make -C ${SRCDIR}/post/build_smesh install

# number of ranks
NX=2
NY=2
NZ=1
NN=$((${NX}*${NY}*${NZ}))

# domain sizes
XS=`grep -w XS conf.h | awk '{print $3}'`
YS=`grep -w YS conf.h | awk '{print $3}'`
ZS=`grep -w ZS conf.h | awk '{print $3}'`
LX=$((${NX}*${XS}))
LY=$((${NY}*${YS}))
LZ=$((${NZ}*${ZS}))

# copy walls
#cp ${SRCDIR}/src/sdf/gx/vessels_mirrored.dat sdf.dat
#cp ${SRCDIR}/src/sdf/gx/vessels_small_mirrored.dat sdf.dat
cp ${SRCDIR}/src/sdf/gx/small.rot.dat sdf.dat
#cp ${SRCDIR}/src/sdf/gx/128.dat sdf.dat

# rbc mesh
cp ${SRCDIR}/src/rbc.off .

# solid mesh
cp ${SRCDIR}/src/data/sphere_R1.ply mesh_solid.ply

# rbc+solid placement
radius=3
fraction=0.5
sc=1
ang=0
plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

# create sbatch script
 echo "#!/bin/bash -l
#
#SBATCH --job-name=vessels
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
