#### googleX small test RBC around cylinder
# cTEST: gx.t1
# export PATH=../tools:$PATH
# rm -rf diag.txt h5 o ply
# cp sdf/gx/small.sdf sdf.dat
# ######cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=8 z=9; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# :
# argp .conf.gx.h \
#    -rbcs -tend=3.0 -part_freq=5000 -walls -wall_creation=1000 \
#    -field_dumps -part_dumps -field_freq=5000 -pushflow > .conf.h
# :
# { make clean && make -j && make -C ../tools; } > /dev/null
# ./udx
# ply2punto ply/rbcs-00001.ply | uscale 100 > ply.out.txt
