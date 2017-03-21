#### Double poiseuille sphere
# sTEST: solid.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# cp bodies/sphere.off rbc.off
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rsph=4 -pin_sph=true      \
#   -tend=2.0 -steps_per_dump=100   \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#
####
# TEST: solid.t1
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.test.h  \
#   -tend=2.0 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000       \
#   -rbcs -rsph=5 -pin_sph=true -dt=1e-3 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# avg_h5.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt
