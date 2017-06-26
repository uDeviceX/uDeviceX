
ic_center () {
    x0=`echo "$LX * 0.5" | bc`
    y0=`echo "$LY * 0.5" | bc`
    z0=`echo "$LZ * 0.5" | bc` 
    echo $x0 $y0 $z0 > ic_solid.txt
}
