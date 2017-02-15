# Usage: sh ./cmp_diag.sh ../mpi-dpd/tests/diag/diag.ref.txt ../mpi-dpd/tests/diag/diag.wrong.txt

f1=$1
f2=$2

t1=/tmp/cmp1.$$.txt
t2=/tmp/cmp2.$$.txt

awk '{print $2}' $f1 > $t1
awk '{print $2}' $f2 > $t2

./fcmp.awk $t1 $t2 && echo "PASS"
