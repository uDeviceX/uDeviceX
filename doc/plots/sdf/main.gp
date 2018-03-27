#! /usr/local/bin/gnuplot -p

set term png
set output "sdf.png"
R=2

set view map
set urange [ -2*R : 2*R ] noreverse nowriteback
set vrange [ -2*R : 2*R ] noreverse nowriteback
set isosamples 100, 100

set xlabel 'x'
set ylabel 'y'

set size ratio -1
set palette defined (-3 "white", 3 "black")

sdf(x,y) = sqrt(x*x + y*y) - R

set contour
set cntrparam levels disc 0
splot '++' u 1:2:(sdf($1,$2)) with pm3d
