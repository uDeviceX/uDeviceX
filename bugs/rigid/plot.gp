f1="cylinder_shear_m1.0/solid_diag_0000.txt"
f2="cylinder_shear_m0.5/solid_diag_0000.txt"
f3="cylinder_shear_m0.1/solid_diag_0000.txt"

set xlabel 't'
set ylabel '\omega'
set grid
plot f1 u 1:10 w l, f2 u 1:10 w l, f3 u 1:10 w l
#plot f1 u 1:10 w lp
#plot f2 u 1:10 w lp

#plot f1 u 1:17 w lp, f2 u 1:17 w lp
