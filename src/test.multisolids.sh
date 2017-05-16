#### spheres in double poiseuille, no contact
# nTEST: multisolid.t1
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt solid_diag.txt
# echo -e "4.1 21.5 8\n4.1 10.5 8\n11.9 16 8" > ic_solid.txt
# cp data/sphere.ply mesh_solid.ply
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -sbounce_back              \
#   -tend=0.5 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -part_dumps     \
#   -wall_creation_stepid=0          \
#   -contactforces=false             \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2test.py bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#

#### spheres in double poiseuille, contact
# nTEST: multisolid.t2
# export PATH=../tools:$PATH
# rm -rf bop h5 diag.txt solid_diag.txt
# echo -e "4.1 21.5 8\n4.1 10.5 8\n11.9 16 8" > ic_solid.txt
# cp data/sphere.ply mesh_solid.ply
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -sbounce_back              \
#   -tend=0.5 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -part_dumps     \
#   -wall_creation_stepid=0          \
#   -contactforces=true              \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2test.py bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#

#### ellipsoids in double poiseuille, contact
# nTEST: multisolid.t3
# export PATH=../tools:$PATH
# rm -rf h5 diag.txt solid_diag.txt
# echo -e "4.1 21.5 8\n4.1 10.5 8\n11.9 16 8" > ic_solid.txt
# cp data/ellipse.ply mesh_solid.ply
# :
# argp .conf.double.poiseuille.h     \
#   -rbcs -sbounce_back              \
#   -tend=0.5 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -part_dumps     \
#   -wall_creation_stepid=0          \
#   -contactforces=true              \
#   -steps_per_hdf5dump=100 > .conf.h
# :
# (make clean && make -j && make -C ../tools) > /dev/null
# ./udx
# bop2test.py bop/solid-00004.bop | awk '{print $1, $2}' | uscale 10 > bop.out.txt
#
