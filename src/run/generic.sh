setup() {
    ( make -C ../tools/rbc install        ) > /dev/null
    ( make -C ../tools install            ) > /dev/null
    ( make -C ../post/build_smesh install ) > /dev/null
}

compile() {
    { make clean && make -j ; } > /dev/null
}

clean () {
    rm -rf diag.txt h5 bop r solid_diag*txt solid-ply
}

geom () {
    NN=$((${NX}*${NY}*${NZ}))
    LX=$((${NX}*${XS}))
    LY=$((${NY}*${YS}))
    LZ=$((${NZ}*${ZS}))
    Domain="XS=$XS YS=$YS ZS=$ZS"
}
