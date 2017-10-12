set term png linewidth 4
set output "diag.png"

set style data line
set key left bottom

set xlabel "time"
set ylabel "velocity"
plot "diag.txt" u 1:3 t "vx", "" u 1:4 t "vy", "" u 1:5 t "vz"