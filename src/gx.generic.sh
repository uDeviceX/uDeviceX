setup() {
    ( make -C ../tools/rbc install        ) > /dev/null
    ( make -C ../tools install            ) > /dev/null
    ( make -C ../post/build_smesh install ) > /dev/null
}

compile() {
    { make clean && make -j ; } > /dev/null
}

clean () {
    rm -rf diag.txt h5 bop ply solid-ply solid_diag*txt
}

cp0() {
    local a=$1 b=$2
    if test ! -r "$a"; then err 'cannot read %s' $a; exit; fi
}

err () {
    printf '%s' 'gx.gneric: '      | cat >&2
    printf "$@"                    | cat >&2
    printf '\n'                    | cat >&2
    exit 1
}


copy() {
    if test $# = 0; then return; fi
    cp0   $1  sdf.dat               ; shift

    if test $# = 0; then return; fi
    cp0   $1  rbc.off               ; shift

    if test $# = 0; then return; fi
    cp0   $1  mesh_solid.ply        ; shift
}

geom () {
    NN=$((${NX}*${NY}*${NZ}))
    LX=$((${NX}*${XS}))
    LY=$((${NY}*${YS}))
    LZ=$((${NZ}*${ZS}))
    Domain="-XS=$XS -YS=$YS -ZS=$ZS"
}
