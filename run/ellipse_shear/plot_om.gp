
f = "< cat solid_diag.txt"

set xlabel "t"
set ylabel "\omega"

gammadot = 0.05
gammadot = 0.0465
a = 2
b = 4
t0 = 11

com_arg(t) = a * b * gammadot * (t-t0) / (a*a + b*b)
sec_sq(x) = 1 / (cos(x)**2)
tan_sq(x) = tan(x)**2
om(t) = a*a * gammadot * sec_sq(com_arg(t)) / ( (b*b + a*a) * (a*a*tan_sq(com_arg(t)) / (b*b) + 1) )

plot f u 1:(-$10) t "\omega_z" w l, f u 1:(om($1)) w l
