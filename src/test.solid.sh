#### Double poiseuille
# TEST: solid.t1
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=8 y=17 z=8
# echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs                            \
#   -tend=2.0 -steps_per_dump=400    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=400 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# avg_h52.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt
