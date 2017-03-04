#!/usr/bin/awk -f

BEGIN {
    Rbem = 0.38; R = Rbem / sc
    
    printf "extent     %s %s %s\n", Lx, Ly, Lz
    printf "N          %d %d %d\n", int(2*Lx), int(2*Ly), int(2*Lz)
    printf "obj_margin 3.0\n"
    printf "\n"
    printf "cylinder axis 0 0 1 point xc yc zc radius %s\n", R
}
