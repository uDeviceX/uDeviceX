setup() {
    ( make -C ../tools/rbc install        ) > /dev/null
    ( make -C ../tools install            ) > /dev/null
    ( make -C ../post/build_smesh install ) > /dev/null
    ( cd ../cmd; make ;                   ) > /dev/null
}

compile() {
    { make clean && u.make -j ; } > /dev/null
}

clean () {
    rm -rf rbcs-ic.txt ic_solid.txt sdf.dat mesh_solid.ply
    rm -rf diag.txt h5 bop r solid_diag*txt solid-ply
}

geom () {
    NN=$((${NX}*${NY}*${NZ}))
    LX=$((${NX}*${XS}))
    LY=$((${NY}*${YS}))
    LZ=$((${NZ}*${ZS}))
    Domain="XS=$XS YS=$YS ZS=$ZS"
}
