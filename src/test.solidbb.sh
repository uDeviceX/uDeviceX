#### pinned sphere, only bb and momentum conservation
#### No DPD forces
#### Note: must modify ic.impl.h manually
# TEST: solidbb.t1
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# :
# argp .conf.double.poiseuille.h                            \
#   -rbcs -rcyl=4 -pin_com=true                             \
#   -tend=5.0 -steps_per_dump=100                           \
#   -hdf5field_dumps -hdf5part_dumps                        \
#   -_aij_out=0.0  -_aij_in=0.0 -_aij_rbc=0.0               \
#   -_gammadpd_out=0.0 -_gammadpd_in=0.0 -_gammadpd_rbc=0.0 \
#   -_hydrostatic_a=0.0                                     \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#

#### pinned ellipse, only bb ans momentum conservation
#### No DPD forces
#### Note: must modify ic.impl.h manually
# sTEST: solidbb.t2
# set -x
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# :
# argp .conf.double.poiseuille.h                            \
#   -rbcs -pin_com=true                                     \
#   -a2_ellipse=16 -b2_ellipse=4                            \
#   -tend=5.0 -steps_per_dump=100                           \
#   -hdf5field_dumps -hdf5part_dumps                        \
#   -_aij_out=0.0  -_aij_in=0.0 -_aij_rbc=0.0               \
#   -_gammadpd_out=0.0 -_gammadpd_in=0.0 -_gammadpd_rbc=0.0 \
#   -_hydrostatic_a=0.0                                     \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# make clean && make -j && make -C ../tools
# ./udx
# read_h5part.m h5/s.h5part | fround.awk -v tol=1 > h5part.out.txt
#
