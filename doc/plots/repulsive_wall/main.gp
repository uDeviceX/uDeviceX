#! /usr/bin/env gnuplot -p

set term png
set output "repulsive_wall.png"
l=1

set xlabel 's'
set ylabel '\phi(s)'

max(a, b) = a < b ? b : a;
phi(s) = max(0, exp(l * (s+1)) - 1)

set xrange [-2:1]
set yrange [-0.5:8]

plot phi(x) w l t ""
