#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh
#
# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

#### RBC in a periodic box
# TEST: diag.t2
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=0.75 y=8 z=12
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# argp .conf.test.h -rbcs -tend=0.5 -steps_per_dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00003.ply | fround.awk -v tol=1 > ply.out.txt

#### RBC initialy rotated
# TEST: rotated.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=0.75 y=8 z=12   c=0.5 s=0.866
# echo $x $y $z    1  0   0 $x              \
#                  0 $c -$s $y              \
#                  0 $s  $c $z              \
#                  0  0   0  1 > rbcs-ic.txt
# argp .conf.test.h -rbcs -tend=0.5 -steps_per_dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00003.ply | fround.awk -v tol=1 > ply.out.txt

#### RBC with wall
# TEST: diag.t3
# export PATH=../tools:$PATH
# cp sdf/wall1/wall.dat                               sdf.dat
# x=0.75 y=8 z=12; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# rm -rf ply h5 diag.txt
# argp .conf.test.h \
#   -rbcs -tend=0.5 -steps_per_dump=300  -walls  -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./test 
# ply2punto ply/rbcs-00003.ply | fround.awk -v tol=1 > ply.out.txt

#### one RBC around cylinder
# TEST: flow.around.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=8 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# argp .conf.around.h \
#    -rbcs -tend=3.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#    -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00001.ply | fround.awk -v tol=1 > ply.out.txt

### two RBCs around cylinder
# TEST: flow.around.t2
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=0.75  y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=0.75 y=13 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.around.h  -rbcs -tend=3.0 -steps_per_dump=5000 \
#        -walls -wall_creation_stepid=1000 \
#        -hdf5field_dumps -hdf5part_dumps  \
#        -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test 
# ply2punto ply/rbcs-00001.ply | fround.awk -v tol=0 > ply.out.txt

### two RBCs around cylinder with one RBC removed by the wall
# TEST: flow.around.t3
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=8    y=8 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# argp .conf.around.h -rbcs -tend=3.0 -steps_per_dump=5000 \
#     -walls -wall_creation_stepid=1000 \
#     -hdf5field_dumps -hdf5part_dumps  \
#     -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./test
# ply2punto ply/rbcs-00001.ply | fround.awk -v tol=1 > ply.out.txt
