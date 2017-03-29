
f = "< cat solid_diag.txt"

set xlabel "t"
set ylabel "\omega"
plot f u 1:10 w lp t "\omega_z"
