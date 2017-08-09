# !/bin/bash

NX=4
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

# Solid position + mesh
x0=`awk -v l=$XS 'BEGIN {print l/2.}'`
y0=`awk -v l=$LY 'BEGIN {print l/2.}'`
z0=`awk -v l=$LZ 'BEGIN {print l/2.}'`
echo $x0 $y0 $z0 > rigs-ic.txt
cp ${GITROOT}/src/data/rig/sphere.ply mesh_solid.ply

# Create walls
echo "extent ${LX} ${LY} ${LZ}
N            100
obj_margin 2.0
plane point 0 $((${LY}-1)) 0 normal 0 -1 0
plane point 0 1 0 normal 0 1 0
plane point 0 0 $((${LZ}-1)) normal 0 0 -1
plane point 0 0 1 normal 0 0 1" > channel.tsdf
${GITROOT}/tsdf/tsdf channel.tsdf sdf.dat sdf.vti

# Restart directory structure
u.strtdir . $NX $NY $NZ

# Run simulation
srun -u -n ${NN} ./udx $NX $NY $NZ 

