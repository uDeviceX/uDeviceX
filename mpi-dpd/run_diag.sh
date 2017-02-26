#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh
#
# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

#### RBC in a periodic box
# TEST: diag.t1
# set -x
# export PATH=../tools:$PATH
# echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# rm -rf ply h5 diag.txt
# argp .conf.test.h -rbcs -tend=0.5 -steps_per_dump=100 > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# awk '{print 100*$2}' diag.txt    | fhash.awk -v tol=2 > diag.out.txt
#

#### Double poiseuille
# TEST: double.poiseuille
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# argp .conf.double.poiseuille.h     \
#   -tend=2.0 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# avg_h52.m h5/flowfields-0013.h5 | fround.awk -v tol=2 > h5.out.txt


#### RBC in a periodic box
# TEST: diag.t2
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=0.75 y=8 z=12
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# argp .conf.test.h -rbcs -tend=0.5 -steps_per_dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# ply2punto ply/rbcs-00003.ply | fround.awk -v tol=1 > ply.out.txt


#### RBC with a wall
# TEST: diag.t3
# export PATH=../tools:$PATH
# cp sdf/wall1/wall.dat                               sdf.dat
# x=0.75 y=8 z=12
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# rm -rf ply h5 diag.txt
# argp .conf.test.h \
#   -rbcs -tend=0.5 -steps_per_dump=300  -walls  -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1 
# ply2punto ply/rbcs-00003.ply | fhash.awk -v tol=1 > ply.out.txt


####
# TEST: diag.t4
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# argp .conf.test.h  \
#   -tend=1.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# avg_h5.m h5/flowfields-0003.h5 | fhash.awk -v tol=1 > h5.out.txt


####
# TEST: diag.t5
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# argp .conf.poiseuille.h \
#   -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# avg_h5.m h5/flowfields-0013.h5 | fhash.awk -v tol=2 > h5.out.txt


####
# TEST: diag.t6
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.poiseuille.h \
#   -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# mid_h5.m h5/flowfields-0026.h5 | fhash.awk -v tol=2 > h5.out.txt


####
# TEST: flow.around.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=8 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# argp .conf.around.h \
#    -rbcs -tend=4.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#    -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# mid_h5.m h5/flowfields-0001.h5 | fhash.awk -v tol=1 > h5.out.txt

### a test case with two RBCs around cylinder
# TEST: flow.around.t2
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=0.75  y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=0.75 y=13 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.around.h  -rbcs -tend=4.0 -steps_per_dump=5000 \
#        -walls -wall_creation_stepid=1000 \
#        -hdf5field_dumps -hdf5part_dumps  \
#        -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1 
# ply2punto ply/rbcs-00001.ply | fhash.awk -v tol=1 > ply.out.txt


### two RBCs around cylinder with one RBC removed by the wall
# TEST: flow.around.t3
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=8    y=8 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# argp .conf.around.h -rbcs -tend=4.0 -steps_per_dump=5000 \
#     -walls -wall_creation_stepid=1000 \
#     -hdf5field_dumps -hdf5part_dumps  \
#     -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 1 1 1
# ply2punto ply/rbcs-00001.ply | fhash.awk -v tol=1 > ply.out.txt
