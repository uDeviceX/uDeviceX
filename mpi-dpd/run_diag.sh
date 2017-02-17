#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh

# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

# TEST: diag.t1
# set -x
# export PATH=../tools:$PATH
# cp .cache.conf.test.h .cache.conf.h
# echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=0.5 -steps_per_dump=100
# awk '{print $2}' diag.txt    | fhash.awk -v tol=2 > diag.out.txt
#
# TEST: diag.t2
# set -x
# export PATH=../tools:$PATH
# cp .cache.conf.test.h .cache.conf.h
# echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=0.5 -steps_per_dump=100
# ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2 > ply.out.txt
#
# TEST: diag.t3
# export PATH=../tools:$PATH
# cp .cache.conf.test.h .cache.conf.h
# cp sdf/wall1/wall.dat                               sdf.dat
# x=0.75 y=8 z=12
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=1.0 -steps_per_dump=300  -walls  -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300
# ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2 > ply.out.txt
#
# TEST: diag.t4
# export PATH=../tools:$PATH
# cp .cache.conf.test.h .cache.conf.h
# cp sdf/wall1/wall.dat sdf.dat
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -tend=1.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300
# avg_h5.m h5/flowfields-0006.h5 | fhash.awk -v tol=2 > h5.out.txt
