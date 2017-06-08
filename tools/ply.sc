#!/usr/bin/awk -f

BEGIN {
    sc = ARGV[1]; shift()
    f  = ARGC > 2 ? ARGV[1] : "-"

#    read_header()
#    read_vert()
#    read_rest()
}

function vertp() {
    return (NF == 3) && NR > 6
}

vertp() {
    x = $1; y = $2; z = $3
    x *= sc; y *= sc; z *= sc
    print x, y, z
}

{ print }

function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }

# TEST: ply.sc.t0
# ply.sc 2 test_data/rbc.ply  > sc.out.ply
#
