#### spheres in double poiseuille, no contact
# TEST: multisolid.t1
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# echo -e "-3.9 5.5 0\n-3.9 -5.5 0\n3.9 0 0" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rsph=4 -pin_com=false     \
#   -tend=2.0 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -contactforces=false             \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#

#### spheres in double poiseuille, contact
# sTEST: multisolid.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# echo -e "-3.9 5.5 0\n-3.9 -5.5 0\n3.9 0 0" > ic_solid.txt
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -rsph=4 -pin_com=false     \
#   -tend=2.0 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -contactforces=true              \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#
