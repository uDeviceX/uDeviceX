#### googleX solvent at rest
# scTEST: gx.t1
# export PATH=../tools:$PATH
# rm -rf diag.txt h5 bop
# cp sdf/gx/small.rot.dat sdf.dat ######## see sdf/gx/README.md
# :
# argp .conf.gx.h \
#    -tend=3.0 -part_freq=5000 -walls -wall_creation=1 \
#    -field_dumps -part_dumps -field_freq=5000 > .conf.h
# :
# { make clean && make -j ; } > /dev/null
# ./udx
# avg_h52.m h5/flowfields-0001.xmf | uscale 0.1 > h5.out.txt
#

#### googleX with body force
# cTEST: gx.t2
# export PATH=../tools:$PATH
# rm -rf diag.txt h5 bop
# cp sdf/gx/small.rot.dat sdf.dat ######## see sdf/gx/README.md
# :
# argp .conf.gx.h \
#    -tend=3.0 -part_freq=5000 \
#    -walls -wall_creation=1   \
#    -pushflow                 \
#    -field_dumps -part_dumps -field_freq=5000 > .conf.h
# :
# { make clean && make -j ; } > /dev/null
# ./udx
# avg_h52.m h5/flowfields-0001.xmf | uscale 0.1 > h5.out.txt
#
