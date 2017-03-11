#### Double poiseuille rbc
# sTEST: solid.t1
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=8 y=17 z=8
# echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# cp bodies/rbc.off rbc.off
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs                            \
#   -tend=1.0 -steps_per_dump=400    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=400 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# ply2punto ply/rbcs-00004.ply | fround.awk -v tol=1 > ply.out.txt
#
#### Double poiseuille sphere
# TEST: solid.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=8 y=17 z=8
# echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# cp bodies/sphere.off rbc.off
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -RBCnv=5220 -RBCnt=10436   \
#   -tend=1.0 -steps_per_dump=400    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=400 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#
#### Double poiseuille rbc rotated crossing domain
# sTEST: solid.t3
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# x=14 y=17 z=8   c=0.5 s=0.866
# echo     1  0   0 $x              \
#          0 $c -$s $y              \
#          0 $s  $c $z              \
#          0  0   0  1 > rbcs-ic.txt
# cp bodies/rbc.off rbc.off
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
# ply2punto ply/rbcs-00009.ply | fround.awk -v tol=1 > ply.out.txt
