#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh

# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

# sTEST: diag.t1
# export PATH=../tools:$PATH
# (cd ../..  && echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=0.5 -shrate=20 -steps_per_dump=100)
# (cd ../..  && awk '{print $2}' diag.txt    | fhash.awk -v tol=2) > diag.out.txt
#
# sTEST: diag.t2
# export PATH=../tools:$PATH
# (cd ../..  && echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=0.5 -shrate=20 -steps_per_dump=100)
# (cd ../..  && ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2) > ply.out.txt
#
# sTEST: diag.t3
# export PATH=../tools:$PATH
# (cd ../..  && cp sdf/cyl1/cyl.dat                                 sdf.dat)
# x=0.75 y=8 z=12
# (cd ../..  && echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=0.5 -shrate=1000 -steps_per_dump=100 -walls  -wall_creation_stepid=1 \
#       -hdf5field_dumps -hdf5part_dumps)
# (cd ../..  && ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2) > ply.out.txt
#
# cTEST: diag.t4
# export PATH=../tools:$PATH
# (cd ../..  && cp sdf/wall1/wall.dat                               sdf.dat)
# x=0.75 y=8 z=12
# (cd ../..  && echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=5.0 -steps_per_dump=300  -walls  -wall_creation_stepid=1 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300)
# (cd ../..  && ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2) > ply.out.txt
