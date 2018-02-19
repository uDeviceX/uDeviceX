# !/bin/bash
#
# Depends on variables:
# ---------------------
# S:    source code dir

# MPI ranks per dim
NX=3
NY=1
NZ=1
NN=$((${NX}*${NY}*${NZ}))

# Domain size
LX=$((${NX}*${XS}))
LY=$((${NY}*${YS}))
LZ=$((${NZ}*${ZS}))

# RBC position
x0=`awk -v l=$LX 'BEGIN {print l/2.}'`
y0=`awk -v l=$LY 'BEGIN {print l/2.}'`
z0=`awk -v l=$LZ 'BEGIN {print l/2.}'`
th=`awk -v th=90 'BEGIN {print th*atan2(0,-1)/180.}'` # for rotation around x-axis
sinth=`awk -v th=$th 'BEGIN {print sin(th)}'`
costh=`awk -v th=$th 'BEGIN {print cos(th)}'`
msinth=`awk -v th=$th 'BEGIN {print -sin(th)}'`
echo 1 0 0 $x0  0 $costh $msinth $y0  0 $sinth $costh $z0  0 0 0 1 > rbcs-ic.txt

# RBC mesh
echo "nv=$nv"
if test $sc -eq 0
then
    cell=$S/data/cells/rbc/${nv}.off
    if [ "$sfree" == "rbc" ]; then
        cell0=$S/data/cells/rbc/${nv}.off
    else
        cell0=$S/data/cells/sph/${nv}.off
    fi
else
    cell=$S/data/cells/rbc/sc/${nv}.off
    if [ "$sfree" == "rbc" ]; then
        cell0=$S/data/cells/rbc/sc/${nv}.off
    else
        cell0=$S/data/cells/sph/sc/${nv}.off
    fi
fi
cp $cell rbc.off
cp $cell0 rbc.stress.free

# Create walls
yw=$((${LY}-1))
n=$((2*${LX}))
echo "extent ${LX} ${LY} ${LZ}
N            $n
obj_margin 2.0
plane point 0 $yw   0 normal 0 -1  0
plane point 0   1   0 normal 0  1  0" > plates.tsdf
${S}/../tsdf/tsdf plates.tsdf sdf.dat sdf.vti

# Restart directory structure
u.strtdir . $NX $NY $NZ

# Create sbatch script
echo "#!/bin/bash -l
#
#SBATCH --job-name=sbatch.sh
#SBATCH --time=24:00:00
#SBATCH --ntasks=${NN}
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out.%j.o
#SBATCH --error=slurm_out.%j.e
#SBATCH --constraint=gpu
#SBATCH --account=ch7

# ======START=====
module load daint-gpu
module load slurm
export CRAY_CUDA_MPS=1

# Run simulation
u.run $NX $NY $NZ ./udx conf.cfg '
glb = {
    kBT = ${kbt}
}
flu = {
    a = [${abb}, ${arb}, ${arr}]
    g = [${gbb}, ${grb}, ${grr}]
}
rbc = {
    gammaC = ${gc}
    kBT = 0.0044433
    kb  = ${kb}
    ks = ${ks}
    x0   = ${X0}
    totArea = ${A0}
    totVolume = ${V0}
}
fsi = {
    a = [0.,     0., 0.    ]
    g = [${grb}, 0., ${grb}]
}
wvel = {
    gdot = ${sh}
}'
# =====END====" > sbatch.sh
