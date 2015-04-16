#!/usr/local/bin/bash

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

if [ $# -ne 5 ]; then
	echo "usage: $0 <# procs along X> <# procs along Y> <# of procs along Z> <tasks per node> <geometry file>"
	exit 1;
fi

mps_per_node=${4}
file=${5}

if [[ ! -f ${file} ]]; then
	echo "Geometry file not found"
	exit 1
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
nz=$3
let tot=nx*ny*nz
let lx=48*nx
let ly=48*ny
let ly2=3*ly/4
let lz=nz*48

cd ../cell-placement
make
./cell-placement ${lx} ${ly2} ${lz}
nrbcs=`wc -l rbcs-ic.txt | awk '{print $1}'`
echo "Generated ${nrbcs} RBCs"
cp rbcs-ic.txt ${wd}/
nctcs=`wc -l ctcs-ic.txt | awk '{print $1}'`
echo "Generated ${nctcs} CTCs"
cp ctcs-ic.txt ${wd}/
cd ../mpi-dpd

#cp one-ic.txt ${wd}/ctcs-ic.txt

fullfile=$(get_abs_filename "${file}")
ln -s ${fullfile} ${wd}/sdf.dat
cp test ${wd}/test

cp ../cuda-rbc/rbc2.atom_parsed ${wd}/../cuda-rbc
cp ../cuda-ctc/sphere.dat ${wd}/../cuda-ctc

echo "********* Global params **********" > ${wd}/params.dat
head -n 50 common.h >> ${wd}/params.dat

find ../ -name "*.h" -o -name "*.cu" -o -name "*.cpp" | xargs tar -cf ${wd}/code.tar

cd ${wd}


if [[ ${mps_per_node} -gt 1 ]]; then
	mps_line="export CRAY_CUDA_MPS=1
export MPICH_ENV_DISPLAY=3
export MPICH_RANK_REORDER_DISPLAY=1
export MPICH_RANK_REORDER_METHOD=2"
fi

let nnodes=tot/mps_per_node

echo "#!/bin/bash -l
#SBATCH --account=s436
#SBATCH --ntasks=${tot}
#SBATCH --nodes=${nnodes}
#SBATCH --time=6:00:00
#SBATCH --signal="USR1"@520
##SBATCH --partition=viz

${mps_line}

export XVELAVG=10
export YVELAVG=3
export ZVELAVG=3
export HEX_COMM_FACTOR=2

aprun -n ${tot} -N ${mps_per_node} ./test ${nx} ${ny} ${nz}
" > iChipCTC${nx}x${ny}x${nz}

sbatch iChipCTC${nx}x${ny}x${nz}

echo "done!"
