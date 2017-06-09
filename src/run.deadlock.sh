#!/usr/bin/sh

setup() {
    export PATH=/usr/lib64/mpich/bin:$PATH
    make -C ../tools/rbc install
    make -C ../tools install
    make -C ../post/build_smesh install
}

pre() {
    XS=20 YS=20 ZS=20
    df=1.0

    D="-XS=$XS -YS=$YS -ZS=$ZS"

    cp cells/rbc.498.off         rbc.off
    cp sdf/pipe_nocenter/sdf.dat sdf.dat
    rm -rf diag.txt h5 bop ply rbc.off solid-ply solid_diag*txt
    
    argp .conf.gx.base.h $D                  \
         -tend=3000.0 -part_freq=1000        \
         -walls -wall_creation=1000          \
         -pushflow -driving_force=$df        \
         -field_dumps -part_dumps -field_freq=1000 > .conf.h
}

compile() {
    { make clean && make -j ; } > /dev/null
}


run() {
    mpirun -n 2 ./udx 1 2 1
}

setup
pre
compile
run
