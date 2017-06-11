# where are googleX geoms?
googlex=/tmp/googlex

ini() {
    export PATH=/usr/lib64/mpich/bin:$PATH
    cp .cache.Makefile.amlucas.panda .cache.Makefile
}

run() { mpiexec -n $NN ./udx $NX $NY $NZ ; }

