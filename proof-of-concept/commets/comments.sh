#!/bin/bash

f=mpi-dpd/common.cu
c=proof-of-concept/commets/comments.awk
t=/tmp/comments.$$.cu

$c $f  > d
diff $f d

for f in `find . -type f -name '*.cu' -or -name '*.h' -or -name '*.cpp'`
do
    $c $f > $t
    cp $t $f
done > d
