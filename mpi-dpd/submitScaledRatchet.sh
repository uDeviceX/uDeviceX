#!/usr/local/bin/bash

if [ $# -ne 5 ]; then
	echo "usage: $0 <# procs along X> <# procs along Y> <# procs along Z> <scale> <sdf filename>"
	exit 1;
fi

if [ ! -f ${5} ]; then
	echo "File doesn't exist"
	exit 1
fi

pattern=/scratch/daint/alexeedm/ctc/funnels
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
nz=$3
scale=$4
let tot=nx*ny*nz
let lx=48*nx
let ly=48*ny
let lx=lx/4
let lz=48*nz

cd ../cell-placement
make
./cell-placement 80 ${ly} ${lz}
nrbcs=`wc -l rbcs-ic.txt | awk '{print $1}'`
echo "Generated ${nrbcs} RBCs"
cp rbcs-ic.txt ${wd}/
nctcs=`wc -l ctcs-ic.txt | awk '{print $1}'`
echo "Generated ${nctcs} CTCs"
cp ctcs-ic.txt ${wd}/
cd ../mpi-dpd

here=`pwd`
ln -s ${here}/${5} ${wd}/sdf.dat
cp test ${wd}/test

cp ../cuda-rbc/cell.dat ${wd}/../cuda-rbc
cp ../cuda-ctc/sphere20.dat ${wd}/../cuda-ctc

echo "********* Global params **********" > ${wd}/params.dat
head -n 35 common.h >> ${wd}/params.dat

find ../ -name "*.h" -o -name "*.cu" -o -name "*.cpp" | xargs tar -cf ${wd}/code.tar

cd ${wd}

#export CRAY_CUDA_MPS=1

echo "#!/bin/bash -l
#SBATCH --account=s436                         
#SBATCH --ntasks=${tot}
#SBATCH --nodes=${tot}
#SBATCH --time=1:00:00
#SBATCH --signal="USR1"@520

export XVELAVG=10
export YVELAVG=3
export ZVELAVG=3
export HEX_COMM_FACTOR=2

aprun -n ${tot} -N 1 ./test ${nx} ${ny} ${nz}
" > SortCells${nx}x${ny}x${nz}

sbatch SortCells${nx}x${ny}x${nz}

echo "done!"
