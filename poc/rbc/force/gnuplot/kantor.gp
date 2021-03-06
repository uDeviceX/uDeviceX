o = "pdf/kantor.pdf"

f0 = "<sort -g data/rbc/a/kantor/" . "0"
f1 = "<sort -g data/rbc/a/kantor/" . "1"
f2 = "<sort -g data/rbc/a/kantor/" . "2"

fe = "<sort -g data/rbc/eq/kantor/" . "01"

r  = "<awk '$1>0' data/rbc/a/curv"

print o

#set term pdfcairo size 3.5in, 3.5in
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
set ylabel "z [L]"           offset  3c
set y2label "energy [T^2/L]" offset -2c

set lmargin 5.5
set rmargin 6.5

H(x,y)=(x + y)/2.0
K(x,y)=x*y

sc = 0.005/2
plot \
     f2 u 1:($2*4)                        axes x1y2 w l lt 3 lw 4 t "initial mesh", \
     r  u 1:((H($3,$4)**2 - K($3,$4))*sc) axes x1y2 w l lt 1 lw 4 t "Helfrich", \
     r                                    w l lt 1 lw 1 t "mesh profile"
