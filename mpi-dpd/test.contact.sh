#!/bin/bash

### contact force: two RBCs in double Poiseuille
# TEST: contact.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
#  x=5 y=17 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=11 y=15 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# argp .conf.double.poiseuille.h -rbcs -tend=2.0 -steps_per_dump=1500 \
#        -hdf5part_dumps   -steps_per_hdf5dump=1500 \
#        -pushtheflow -doublepoiseuille \
#        -contactforces > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00002.ply | fhash.awk -v tol=1 > ply.out.txt

### no contact force: two RBCs in double Poiseuille
# sTEST: no.contact.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
#  x=4 y=17 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=12 y=15 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# argp .conf.double.poiseuille.h -rbcs -tend=2.0 -steps_per_dump=1500 \
#        -hdf5part_dumps -steps_per_hdf5dump=1500 \
#        -pushtheflow -doublepoiseuille \
#                      > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00002.ply | fhash.awk -v tol=1 > ply.out.txt
