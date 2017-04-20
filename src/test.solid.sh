#### Couette pinned sphere
# TEST: solid.t1
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# echo -e "16 16 16" > ic_solid.txt
# :
# argp .conf.test.h  \
#   -tend=2.0 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000       \
#   -rbcs -rsph=5 -pin_com=true -dt=1e-3 -shear_z > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# avg_h5.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt

#### Double poiseuille pinned sphere
# sTEST: solid.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rsph=4 -pin_com=false      \
#   -tend=2.0 -steps_per_dump=100   \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#

#### Double poiseuille pinned cylinder
# sTEST: solid.t3
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt debug.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rcyl=4 -pin_com=true      \
#   -tend=2.0 -steps_per_dump=100   \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#

#### Couette pinned cylinder
# sTEST: solid.t4
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# cp sdf/yplates1/yplates.dat sdf.dat
# echo -e "16 16 16" > ic_solid.txt
# :
# argp .conf.test.h  \
#   -tend=2.0 -steps_per_dump=1000 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=1000       \
#   -rbcs -rcyl=5 -pin_com=true -dt=1e-3 -shear_y > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# avg_h5.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt

#### Double poiseuille pinned ellipse
# sTEST: solid.t5
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt debug.txt
# echo -e "8 16 8" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs  -pin_com=true             \
#   -a2_ellipse=16 -b2_ellipse=4     \
#   -tend=2.0 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#
