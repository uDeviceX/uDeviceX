ini() {
    module load cray-hdf5-parallel cray-mpich cudatoolkit
}

run () {
    n=gx
    
    cat <<-EOF > runme.sh
	#!/bin/bash -l
    #SBATCH --partition=low
	#SBATCH --job-name=$n
	#SBATCH --time=00:10:00
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=1
	#SBATCH --output=output.txt
	#SBATCH --error=error.txt
	#SBATCH --constraint=gpu
	:
    module load cray-hdf5-parallel cray-mpich cudatoolkit
	srun --export ALL ./udx
EOF
    sbatch runme.sh
}
