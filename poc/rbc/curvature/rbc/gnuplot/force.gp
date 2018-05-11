f = "data/force/b"
o = "pdf/force/b.pdf"
#set y2range [-5:5]
set y2range [-35:35]

set term pdfcairo monochrome size 3.5in, 3.5in
print o
set output o

set size square
set xrange [0:1]
set yrange [-0.5:0.5]

set key left bottom

set xlabel "r [L]"
set xtics  0.5
set ytics  0.4 nomirror
set noy2tics
set ylabel "z [L]"            offset  3c

set lmargin 5.5
set rmargin 6.5

plot f                              w l lt 1 lw 1 t "",       \
     "" u 1:3             axes x1y2 w l lt 1 lw 3 t "energy", \
     "" u 1:($4/50)       axes x1y2 w l      lw 3 t "force"

set output