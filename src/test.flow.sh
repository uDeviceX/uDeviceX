#!/bin/bash

# Run from this directory:
#  > atest test.flow.sh
#
# To update the test change TEST to cTEST and run
#  > atest test.flow.sh

#### Double poiseuille
# nTEST: flow.t1
# rm -rf bop h5 diag.txt rbc.off
# :
# export PATH=../tools:$PATH
# cp cells/rbc.498.off  rbc.off
# :
# argp .conf.double.poiseuille.h      \
#    -rsph=4                          \
#    -tend=2.01 -part_freq=4000  \
#    -pushflow -doublepoiseuille   \
#    -field_dumps -part_dumps     \
#    -field_freq=4000 -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h52.m h5/flowfields-0001.h5 | uscale 0.1 > h5.out.txt

#### Plates : shear
# nTEST: flow.t2
# rm -rf bop h5 diag.txt rbc.off
# :
# export PATH=../tools:$PATH
# cp cells/rbc.498.off  rbc.off
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.test.h                                                 \
#    -rsph=4                                                        \
#    -tend=1.0 -part_freq=300 -walls -wall_creation=100 \
#    -field_dumps -part_dumps -field_freq=300           \
#    -shear_z > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0003.h5 | uscale 0.1 > h5.out.txt

#### Plates : poiseuille
# nTEST: flow.t3
# rm -rf bop h5 diag.txt rbc.off
# :
# export PATH=../tools:$PATH
# cp cells/rbc.498.off  rbc.off
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.poiseuille.h                                           \
#    -rsph=4                                                        \
#    -tend=2.0 -part_freq=300 -walls -wall_creation=100 \
#    -field_dumps -part_dumps -field_freq=300           \
#    -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0013.h5 > h5.out.txt

#### flow around cylinder
# nTEST: flow.t4
# rm -rf bop h5 diag.txt rbc.off
# :
# export PATH=../tools:$PATH
# cp cells/rbc.498.off rbc.off
# cp sdf/cyl1/cyl.dat sdf.dat
# :
# argp .conf.poiseuille.h                                           \
#    -rsph=4                                                        \
#    -tend=4.0 -part_freq=300 -walls -wall_creation=100 \
#    -field_dumps -part_dumps -field_freq=300 -pushflow > .conf.h
# :
# (make clean && make -j) > /dev/null
# ./udx
# mid_h5.m h5/flowfields-0026.h5 > h5.out.txt


