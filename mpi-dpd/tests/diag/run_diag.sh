#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh

# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

# TEST: diag.t1
# export PATH=../tools:$PATH
# (cd ../..  && echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=0.5 -shrate=20 -steps_per_dump=100)
# (cd ../..  && awk '{print $2}' diag.txt    | fhash.awk -v tol=2) > diag.out.txt
#
# TEST: diag.t2
# export PATH=../tools:$PATH
# (cd ../..  && echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt)
# (cd ../..  && make clean && make -j && make -C ../tools)
# (cd ../..  && ./test 1 1 1 -rbcs -tend=0.5 -shrate=20 -steps_per_dump=100)
# (cd ../..  && ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2) > ply.out.txt
