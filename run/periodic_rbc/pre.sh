#!/bin/bash

ic () {
    radius=6.0
    fraction=0.2
    sc=1.0
    ang=0.785
    plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt
    rm ic_solid.txt
    cp ${GITROOT}/src/data/cells/rbc.498.off rbc.off
}

restart_dir () {
    # Restart directory structure
    u.strtdir . $NX $NY $NZ
}
