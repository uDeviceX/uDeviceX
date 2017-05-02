#!/bin/bash

# Run from this directory:
#  > atest test.flow.sh
#
# To update the test change TEST to cTEST and run
#  > atest test.flow.sh
# add crap from test_data/* to git

#### Double poiseuille
# sTEST: double.poiseuille
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# :
# argp .conf.double.poiseuille.h      \
#    -rsph=4 -pin_com=true            \
#    -tend=2.01 -steps_per_dump=4000  \
#    -pushtheflow -doublepoiseuille   \
#    -hdf5field_dumps -part_dumps     \
#    -steps_per_hdf5dump=4000 -pushtheflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h52.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt

#### Plates : shear
# sTEST: diag.t4
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.test.h  \
#    -rsph=4 -pin_com=true                                          \
#    -tend=1.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#    -hdf5field_dumps -part_dumps -steps_per_hdf5dump=300           \
#    -shear_z > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0003.h5 | fround.awk -v tol=1 > h5.out.txt

#### Plates : poiseuille
# sTEST: diag.t5
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.poiseuille.h \
#    -rsph=4 -pin_com=true                                         \
#   -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -part_dumps -steps_per_hdf5dump=300          \
#   -pushtheflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0013.h5 | fround.awk -v tol=2 > h5.out.txt

#### Cylinder
# sTEST: diag.t6
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# :
# argp .conf.poiseuille.h \
#    -rsph=4 -pin_com=true                                         \
#   -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# mid_h5.m h5/flowfields-0026.h5 | fhash.awk -v tol=2 > h5.out.txt


