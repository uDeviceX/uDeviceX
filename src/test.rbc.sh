

#### RBC in a periodic box
# nTEST: diag.t2
# export PATH=../tools:$PATH
# rm -rf diag.txt h5 o ply
# x=0.75 y=8 z=12; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# :
# argp .conf.couette.h -rbcs -tend=0.5 -part_freq=300 > .conf.h
# :
# { make clean && make -j && make -C ../tools; } > /dev/null
# ./udx
# ply2punto ply/rbcs-00003.ply | uscale 100 > ply.out.txt
