
#### contact force: two RBCs in double Poiseuille
# nTEST: contact.t1
# export PATH=../tools:$PATH
# rm -rf diag.txt h5 o ply
#  x=5 y=17 z=8; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=11 y=15 z=8; echo 1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# :
# argp .conf.double.poiseuille.h -rbcs -tend=2.0 -part_freq=1500 \
#        -part_dumps   -field_freq=1500 \
#        -pushflow -doublepoiseuille \
#        -contactforces > .conf.h
# :
# { make clean && make ranks && make -j && make -C ../tools; } > /dev/null
# udirs sr/p
# ./u
# ply2punto ply/rbcs-00002.ply | uscale 100 > ply.out.txt
