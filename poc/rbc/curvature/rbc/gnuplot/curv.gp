load "gnuplot/a.gp"

set term pdfcairo monochrome size 3.5in, 3.5in
print o, f
set output o

set size square
set xrange [0:1]
set yrange [-0.5:0.5]
set y2range [-8.5:8.5]

set key left bottom

set xlabel "r [L]"
set xtics  0.5
set ytics  0.4 nomirror
set y2tics 5.0 nomirror
set ylabel "z [L]"            offset  3c
set y2label "curvature [1/L]" offset -2c

set lmargin 5.5
set rmargin 6.5

plot f                              w l lt 1 lw 1 t "",          \
     "" u 1:3             axes x1y2 w l      lw 3 t "c_1",       \
     "" u 1:4             axes x1y2 w l      lw 3 t "c_2",       \
     "" u 1:($3 + $4)     axes x1y2 w l lt 1 lw 6 t "c_1 + c_2", \

set output