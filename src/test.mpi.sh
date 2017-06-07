#!/bin/bash

#### RBC in a periodic box
# sTEST: mpi.t1
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# echo 1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# rm -rf diag.txt h5 bop ply
# :
# argp .conf.couette.h -rbcs -tend=0.5 -part_freq=100 > .conf.h
# :
# { make clean && make -j; } > /dev/null
# mpirun -n 2 ./udx 2 1 1
# awk '{print 10000*$2}' diag.txt > diag.out.txt

#### double Poiseuille
# nTEST: mpi.t2
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf diag.txt h5 bop ply
# :
# argp .conf.double.poiseuille.h                \
#   -tend=2.1 -part_freq=100  -field_freq=300   \
#   -pushflow -doublepoiseuille                 \
#   -field_dumps -part_dumps                    \
#   -pushflow > .conf.h
# :
# { make clean && make -j; } > /dev/null
# mpirun -n 2 ./udx 2 1 1
# avg_h52.m h5/f.0013.h5 | uscale 10 > h5.out.txt

####
# sTEST: mpi.t3
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf diag.txt h5 o ply
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=8 z=9; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# :
# argp .conf.around.h \
#    -acyl            \
#    -rbcs -tend=4.0 -part_freq=5000 -walls -wall_creation=1000 \
#    -field_dumps -part_dumps -field_freq=5000 -pushflow > .conf.h
# :
# x=3 y=1 z=1
# { make clean && make ranks x=$x y=$y z=$z && make -j mpi; } > /dev/null
# udirs $x $y $z sr/p
# sh run
# mid_h5.m h5/f.0001.h5 > h5.out.txt
#

#### Poiseuille
# sTEST: mpi.t4
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf diag.txt h5 o ply
# cp sdf/wall1/wall.dat sdf.dat
# :
# argp .conf.poiseuille.h \
#   -awall                \
#   -tend=4.0 -part_freq=600 -walls -wall_creation=100 \
#   -field_dumps -part_dumps -field_freq=600 -pushflow > .conf.h
# :
# x=1 y=1 z=2
# { make clean && make ranks x=$x y=$y z=$z && make -j mpi; } > /dev/null
# udirs $x $y $z sr/p
# sh run
# avg_h5.m h5/f.0013.h5 | uscale 100 > h5.out.txt
#
