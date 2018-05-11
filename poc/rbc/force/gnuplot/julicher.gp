o = "pdf/gompper.pdf"

f0 = "<sort -g data/rbc/a/gompper/" . "0"
f1 = "<sort -g data/rbc/a/gompper/" . "1"
f2 = "<sort -g data/rbc/a/gompper/" . "2"
r  = "<awk '$1>0' data/rbc/a/curv"

print o

set term pdfcairo size 3.5in, 3.5in
set output o

set size square
set xrange [0:1]
set yrange [0:1]
set y2range [0:0.15]

set key left top

set xlabel "r [L]"
set xtics  0.5
set ytics  0.4 nomirror
set y2tics 5.0 nomirror
set ylabel "z [L]"            offset  3c
set y2label "energy [T^2/L]" offset -2c

set lmargin 5.5
set rmargin 6.5

sc = 0.0015
plot r                                  w l lt 1 lw 1 t "",  \
     f2 u 1:($2*16)           axes x1y2 w l lt 3 lw 2 t "2", \
     f1 u 1:($2*4)            axes x1y2 w l lt 2 lw 2 t "1", \
     f0                       axes x1y2 w l lt 1 lw 2 t "0", \
     r  u 1:(($3 + $4)**2*sc) axes x1y2 w l lt 1 lw 4 t "reference"
