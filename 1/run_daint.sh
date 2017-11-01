# !/bin/bash

# MPI ranks per dim
NX=`awk -v sc=${L} 'BEGIN {print 3*sc}'`  #4,6: from Gerris!
NY=`awk -v sc=${L} 'BEGIN {print 1*sc}'`
NZ=`awk -v sc=${L} 'BEGIN {print 1*sc}'`
echo "Nx=$NX Ny=$NY Nz=$NZ"

# Domain size
XS=`grep -w XS conf.h | awk '{print $3}'`
YS=`grep -w YS conf.h | awk '{print $3}'`
ZS=`grep -w ZS conf.h | awk '{print $3}'`
NN=$((${NX}*${NY}*${NZ}))
LX=$((${NX}*${XS}))
LY=$((${NY}*${YS}))
LZ=$((${NZ}*${ZS}))
echo $LX $LY $LZ

# Solid position
x0=`awk -v l=$XS 'BEGIN {print l/2.}'`
y0=`awk -v l=$LY -v p=${iyp} -v w=${L} 'BEGIN {print w+p*(l-2.*w)/2.}'`  #assumes p=[0,1]
z0=`awk -v l=$LZ 'BEGIN {print l/2.}'`
echo $x0 $y0 $z0 > rigs-ic.txt
# mesh
cp ${GITROOT}/src/data/rig/spheres/sphereL2_D${D}.ply mesh_solid.ply

# Create walls
yw=$((${LY}-${L}))
zw=$((${LZ}-${L}))
n=$((2*${LX}))
echo "extent ${LX} ${LY} ${LZ}
N            $n
obj_margin 2.0
plane point 0 $yw   0 normal 0 -1  0
plane point 0  $L   0 normal 0  1  0
plane point 0   0 $zw normal 0  0 -1
plane point 0   0  $L normal 0  0  1" > channel.tsdf
${GITROOT}/tsdf/tsdf channel.tsdf sdf.dat sdf.vti

# Restart directory structure
u.strtdir . $NX $NY $NZ

# Create+run sbatch script
echo "#!/bin/bash -l
#
#SBATCH --job-name=chsolL${L}D${D}
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

# Save original files 
find . > setup.files

# Submit on daint
sbatch sbatch_gen.sh
#srun -u -n ${NN} ./udx ${NX} ${NY} ${NZ}
