echo "0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1" > rbcs-ic.txt
make clean && make -j
./test 1 1 1 -rbcs -tend=5000e-4 -shrate=8 -steps_per_dump=100
cp diag.txt tests/diag/diag.test.txt
(cd ../tools/ && sh ./cmp_diag.sh ../mpi-dpd/tests/diag/diag.ref.txt ../mpi-dpd/tests/diag/diag.test.txt)
