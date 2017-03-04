#!/usr/bin/awk -f

BEGIN {
    Nx = int(2*Lx); Ny=int(2*Ly); Nz=int(2*Lz)

    printf "extent     %g %g %g\n", Lx, Ly, Lz
    printf "N          %d %d %d\n", Nx, Ny, Nz
    printf "obj_margin 3.0\n"
    printf "\n"
    # possible values for axis: XY, XZ, YZ
    #        1    2  3     4  5  6  7      8      9      10    11    12      13  14
    rx=8.5 / 32 * Lx
    ry=12  / 56 * Ly
    printf "egg axis XY point xc yc zc radius      %g   %g    angle 0   eggness 0.3\n", rx, ry
}
