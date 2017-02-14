#!/bin/bash

dpd_dir=$HOME/rbc_shear_tags/mpi-dpd
rbc_dir=$HOME/rbc_shear_tags/cuda-rbc

sed -i "/XSIZE_SUBDOMAIN/c\    XSIZE_SUBDOMAIN = 16," $dpd_dir/common.h
sed -i "/YSIZE_SUBDOMAIN/c\    YSIZE_SUBDOMAIN = 16,"  $dpd_dir/common.h
sed -i "/ZSIZE_SUBDOMAIN/c\    ZSIZE_SUBDOMAIN = 16,"  $dpd_dir/common.h
echo "0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1" > $dpd_dir/rbcs-ic.txt

st=6
fin=6

mkdir -p simulations

for i in $(seq $st $fin); do
	echo "Running case $i"

	# read parameters
	aij1=$(     awk -v n=$i 'NR==n {print $1}'  points.txt)
	aij2=$(     awk -v n=$i 'NR==n {print $2}'  points.txt)
	nd=$(       awk -v n=$i 'NR==n {print $3}'  points.txt)
	gammadpd1=$(awk -v n=$i 'NR==n {print $4}'  points.txt)
	gammadpd2=$(awk -v n=$i 'NR==n {print $5}'  points.txt)
	kBT=$(      awk -v n=$i 'NR==n {print $6}'  points.txt)
	shrate=$(   awk -v n=$i 'NR==n {print $7}'  points.txt)
	gammaC=$(   awk -v n=$i 'NR==n {print $8}'  points.txt)
	kb=$(       awk -v n=$i 'NR==n {print $9}'  points.txt)
	p=$(        awk -v n=$i 'NR==n {print $10}' points.txt)
	x0=$(       awk -v n=$i 'NR==n {print $11}' points.txt)

	# recompile
	(
	cd $dpd_dir

	sed -i "/const int numberdensity/c\const int numberdensity = $nd * (RC_FX*RC_FX*RC_FX); \/\/ default: 3" common.h
	sed -i "/const float kBT/c\const float kBT = $kBT * kBT2D3D / (RC_FX*RC_FX); \/\/ default: 1" common.h

	sed -i "/const float gammadpd\[/c\    const float gammadpd[4] = {$gammadpd1, $gammadpd2, $gammadpd1, $gammadpd1}; \/\/ default: 4.5" dpd-forces.cu
	sed -i "/const float aij\[/c\    const float aij[4] = {$aij1 / RC_FX, $aij2 / RC_FX, $aij1 / RC_FX, $aij1 / RC_FX}; \/\/ default: 75*kBT/numberdensity -- Groot and Warren (1997)" dpd-forces.cu

	make -j slevel=-1
	)

	# prepare working directory
	dir=simulations/aij1_${aij1}_aij2_${aij2}_nd_${nd}_gammadpd1_${gammadpd1}_gammadpd2_${gammadpd2}_kBT_${kBT}_shrate_${shrate}_gammaC_${gammaC}_kb_${kb}_p_${p}_x0_${x0}
	mkdir -p $dir
	(
	cd $dir

	mkdir -p cuda-rbc
	cp $rbc_dir/rbc.dat cuda-rbc

	mkdir -p mpi-dpd
	cd mpi-dpd
	cp $dpd_dir/test .
	cp $dpd_dir/rbcs-ic.txt .
	cp $HOME/scripts/rbc/compute_freq_standalone_Tran-Son-Tay.py .
	cp $HOME/scripts/rbc/plyfile.py .
	awk -v n=$i 'NR==n' ../../../points.txt > params.txt

	st_p_d=$(awk -v sh=$shrate 'BEGIN {print int(1000/sh)}')
	tend=$(awk -v sh=$shrate 'BEGIN {print int(2000/sh)}')

	rm -f runme.sh
	touch runme.sh
	echo "#!/bin/bash -l" >> runme.sh
	echo "#SBATCH --job-name=sh_${i}" >> runme.sh
	echo "#SBATCH --time=12:00:00" >> runme.sh
	echo "#SBATCH --nodes=1" >> runme.sh
	echo "#SBATCH --ntasks-per-node=1" >> runme.sh
	echo "#SBATCH --output=sh.%j.o" >> runme.sh
	echo "#SBATCH --error=sh.%j.e" >> runme.sh
	echo "#SBATCH --account=ch7" >> runme.sh
	echo "#SBATCH --constraint=gpu" >> runme.sh
	echo "export HEX_COMM_FACTOR=3" >> runme.sh
	echo "srun -n 1 --export ALL ./test 1 1 1 -tend=$tend -rbcs -steps_per_dump=$st_p_d -shrate=$shrate -RBCtotArea=124 -RBCtotVolume=90 -RBCka=4900 -RBCkb=$kb -RBCkd=100 -RBCkv=5000 -RBCgammaC=$gammaC -RBCx0=$x0 -RBCp=$p -hdf5field_dumps -steps_per_hdf5dump=$st_p_d -hdf5part_dumps" >> runme.sh

	sbatch runme.sh
	)
done
