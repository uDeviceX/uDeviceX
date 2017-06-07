#!/bin/bash

# Run from this directory:
#  > atest test.flow.sh
#
# To update the test change TEST to cTEST and run
#  > atest test.flow.sh
# add crap from test_data/* to git

#### Double poiseuille
# nTEST: flow.t1
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt
# :
# argp .conf.double.poiseuille.h      \
#    -rsph=4                          \
#    -tend=2.01 -steps_per_dump=4000  \
#    -pushflow -doublepoiseuille   \
#    -field_dumps -part_dumps     \
#    -steps_per_hdf5dump=4000 -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h52.m h5/flowfields-0001.h5 | uscale 0.1 > h5.out.txt

#### Plates : shear
# nTEST: flow.t2
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.test.h                                                 \
#    -rsph=4                                                        \
#    -tend=1.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#    -field_dumps -part_dumps -steps_per_hdf5dump=300           \
#    -shear_z > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0003.h5 | uscale 0.1 > h5.out.txt

#### Plates : poiseuille
# nTEST: flow.t3
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.poiseuille.h                                           \
#    -rsph=4                                                        \
#    -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#    -field_dumps -part_dumps -steps_per_hdf5dump=300           \
#    -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0013.h5 > h5.out.txt

#### flow around cylinder
# nTEST: flow.t4
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# :
# argp .conf.poiseuille.h                                           \
#    -rsph=4                                                        \
#    -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#    -field_dumps -part_dumps -steps_per_hdf5dump=300 -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# mid_h5.m h5/flowfields-0026.h5 > h5.out.txt


