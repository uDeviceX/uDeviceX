# !/bin/bash

NX=1
NY=1
NZ=1

# Domain size
XS=`grep -w XS conf.h | awk '{print $3}'`
YS=`grep -w YS conf.h | awk '{print $3}'`
ZS=`grep -w ZS conf.h | awk '{print $3}'`
NN=$((${NX}*${NY}*${NZ}))
LX=$((${NX}*${XS}))
LY=$((${NY}*${YS}))
LZ=$((${NZ}*${ZS}))
echo $LX $LY $LZ

# RBCs position + mesh
# rbc+solid placement
radius=6.0
fraction=0.2
sc=1.0
ang=0.785
plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt
rm ic_solid.txt
cp ${GITROOT}/src/data/cells/rbc.498.off rbc.off

# Create walls
yw=$((${LY}-1))
zw=$((${LZ}-1))
n=$((2*${LX}))
echo "extent ${LX} ${LY} ${LZ}
N            $n
obj_margin 2.0
plane point 0 $yw   0 normal 0 -1 0
plane point 0   1   0 normal 0 1 0
plane point 0   0 $zw normal 0 0 -1
plane point 0   0   1 normal 0 0 1" > channel.tsdf
${GITROOT}/tsdf/tsdf channel.tsdf sdf.dat sdf.vti

# Restart directory structure
u.strtdir . $NX $NY $NZ

# Create+run sbatch script
echo "#!/bin/bash -l
#
#SBATCH --job-name=test-color
#SBATCH --time=24:00:00
#SBATCH --ntasks=${NN}
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out.%j.o
#SBATCH --constraint=gpu
#SBATCH --error=slurm_out.%j.e
#SBATCH --account=ch7

# ======START=====
module load daint-gpu
module load slurm
export CRAY_CUDA_MPS=1

# Run simulation
srun -u -n ${NN} ./udx $NX $NY $NZ 

# =====END====" > sbatch_gen.sh
 
# Submit on daint
sbatch sbatch_gen.sh
#srun -u -n ${NN} ./udx ${NX} ${NY} ${NZ}
