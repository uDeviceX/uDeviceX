#!/bin/bash

set -eu

c=`pwd`
(
    d=$HOME/EXTERN/SDAINT/SYNC/resolution
    #cd $d/level_1_myaij_0_ha_0.00938_geom_cir_gammadpd_15_aij_2_Nv_162_cshape_sphere_nd_10_kb_5/ply
    cd $d/level_2_myaij_0_ha_0.00938_geom_cir_gammadpd_15_aij_2_Nv_362_cshape_sphere_nd_10_kb_5
    
    find . -name 'rbcs-*.ply' | sort -g > v.visit

    vpython.sh -i $c/rbc.py
    
    mv *.png $c/
)

feh -F -d visit*.png
