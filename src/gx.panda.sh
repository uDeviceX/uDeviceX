# where are googleX geoms?
googlex=/tmp/googlex

ini() {
    export PATH=/usr/lib64/mpich/bin:$PATH
}

run() { mpiexec -n $NN ./udx $NX $NY $NZ ; }
