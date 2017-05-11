#### Couette pinned sphere
# nTEST: mbounce.t1
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# echo -e "16 16 16" > ic_solid.txt
# :
# argp .conf.test.h                                                \
#   -tend=2.01 -steps_per_dump=500 -walls -wall_creation_stepid=0  \
#   -hdf5field_dumps -part_dumps  -steps_per_hdf5dump=1000         \
#   -rbcs -sbounce_back -pin_com -dt=1e-3 -shear_z > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2txt bop/solid-00004.bop | awk '{print $1, $2}' > bop.out.txt

#### Double poiseuille non pinned sphere
# nTEST: mbounce.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h      \
#   -rbcs  -sbounce_back              \
#   -tend=0.5 -steps_per_dump=100     \
#   -pushtheflow -doublepoiseuille    \
#   -hdf5field_dumps -part_dumps      \
#   -wall_creation_stepid=0           \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2txt bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#
