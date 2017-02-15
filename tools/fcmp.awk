#!/usr/bin/awk -f

# "Fuzzy" compares two columns of data in two files independent of order
#
# Usage:
# ./fcmp.awk <file1> <file2>
# ./fcmp.awk <file1> <file2> && echo "good"
# ./fcmp.awk <file1> <file2> || echo "bad"

# TEST: fcmp.t1
# seq 10      > f1
# seq 10 -1 1 > f2
# ./fcmp.awk  f1 f2 && echo t1 > true.out.txt

# TEST: fcmp.t2
# seq 10      > f1
# seq  9 -1 0 > f2
# ./fcmp.awk  f1 f2 || echo t2 > true.out.txt

# TEST: fcmp.t3
# echo 0.001001      > f1
# echo 1e-3          > f2
# ./fcmp.awk  f1 f2 && echo t3 > true.out.txt

# TEST: fcmp.t4
# printf 0.001\\n1e-2\\n > f1
# printf 1e-3\\n0.01\\n  > f2
# ./fcmp.awk  f1 f2 && echo t4 > true.out.txt

function hash(e,  h) {
    h = sprintf("%.2g", e)
    # printf "hash[%s] = %s\n", e, h | "cat 2>&1"
    return h
}

{
    # for both files
    h = hash($1)
}

FNR == NR { # for the first file
    tbl[h] ++
}

FNR != NR { # the second file
    tbl[h] --
}

function err() {
    printf "(fcmp.awk) error on hash: %s\n", h | "cat 2>&1"
    exit(-1)
}

END {
    for (h in tbl)
	if (tbl[h] != 0) err()
    printf "(fcmp.awk) fuzzy-same\n" | "cat 2>&1"
}
