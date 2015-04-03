#!/usr/local/bin/bash

if [ $# -ne 2 ]; then
	echo "usage: $0 <# procs along X> <# procs along Y>"
	exit 1;
fi

pattern=/scratch/daint/alexeedm/ctc/ichip
i=-1

while [ 1 ]
do
	let i+=1
	wd=${pattern}${i}
	[[ ! -e ${wd} ]] && break
done

echo "Working directory is ${wd}"

mkdir -p ${wd}
mkdir -p ${wd}/../cuda-rbc
mkdir -p ${wd}/../cuda-ctc

nx=$1
ny=$2
let tot=nx*ny
let lx=48*nx
let ly=48*ny
let lx4=lx/4

cd ../cell-placement
make
./cell-placement ${lx} ${ly} 48
nrbcs=`wc -l rbcs-ic.txt | awk '{print $1}'`
echo "Generated ${nrbcs} RBCs"
cp rbcs-ic.txt ${wd}/
nctcs=`wc -l ctcs-ic.txt | awk '{print $1}'`
echo "Generated ${nctcs} CTCs"
cp ctcs-ic.txt ${wd}/
cd ../mpi-dpd

here=`pwd`
ln -s ${here}/eggs/${nx}x${ny}.dat ${wd}/sdf.dat
cp test ${wd}/test

cp ../cuda-rbc/rbc2.atom_parsed ${wd}/../cuda-rbc
cp ../cuda-ctc/sphere.dat ${wd}/../cuda-ctc

echo "********* Global params **********" > ${wd}/params.dat
head -n 35 common.h >> ${wd}/params.dat

find ../ -name "*.h" -o -name "*.cu" -o -name "*.cpp" | xargs tar -cf ${wd}/code.tar

cd ${wd}

#export CRAY_CUDA_MPS=1

echo "#!/bin/bash -l
#SBATCH --account=s448               
#SBATCH --ntasks=${tot}
#SBATCH --nodes=${tot}
#SBATCH --time=3:00:00
#SBATCH --signal="USR1"@520

export XVELAVG=10
export YVELAVG=3
export ZVELAVG=3
export HEX_COMM_FACTOR=2

aprun -n ${tot} -N 1 ./test ${nx} ${ny} 1
" > iChip1${nx}x${ny}

sbatch iChip1${nx}x${ny}

echo "done!"
