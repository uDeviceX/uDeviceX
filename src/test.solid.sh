#### Couette pinned sphere
# sTEST: solid.t1
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# echo -e "16 16 16" > ic_solid.txt
# :
# argp .conf.test.h  \
#   -tend=2.01 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -part_dumps  -steps_per_hdf5dump=1000           \
#   -rbcs -rsph=5 -pin_com=true -dt=1e-3 -shear_z > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0001.h5 | sed -n '4,29p' | uscale 0.5 > h5.out.txt

#### Double poiseuille pinned sphere
# nTEST: solid.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h      \
#   -rbcs -rsph=4 -pin_com=false      \
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

#### Double poiseuille pinned cylinder
# nTEST: solid.t3
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt debug.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rcyl=4 -pin_com=true      \
#   -tend=0.5 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -part_dumps     \
#   -wall_creation_stepid=0          \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2txt bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#

#### Couette pinned cylinder
# sTEST: solid.t4
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt
# cp sdf/yplates1/yplates.dat sdf.dat
# echo -e "16 16 16" > ic_solid.txt
# :
# argp .conf.test.h  \
#   -tend=2.01 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -part_dumps -steps_per_hdf5dump=1000            \
#   -rbcs -rcyl=5 -pin_com=true -dt=1e-3 -shear_y > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# avg_h5.m h5/flowfields-0001.h5 | sed -n '4,29p' | uscale 0.5 > h5.out.txt

#### Double poiseuille pinned ellipse
# nTEST: solid.t5
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 bop diag.txt solid_diag.txt debug.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs  -pin_com=true             \
#   -a2_ellipse=16 -b2_ellipse=4     \
#   -tend=0.5 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -part_dumps     \
#   -wall_creation_stepid=0          \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2txt bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#
