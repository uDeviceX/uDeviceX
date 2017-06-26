
f = "< sed -n '1,3000p' results/test/solid_diag.txt"

set xlabel "t"
set ylabel "\omega"

gammadot = 0.05
gammadot = 0.0465

gammadot = 0.0325

a = 2
b = 4
t0 = 30
G=gammadot

com_arg(t, a_, b_, G_) = a_ * b_ * G_ * t / (a_*a_ + b_*b_)
sec_sq(x) = 1 / (cos(x)**2)
tan_sq(x) = tan(x)**2

om_th(t, a_, b_, G_) = a_*a_ * G_ * sec_sq(com_arg(t, a_, b_, G_)) / ( (b_*b_ + a_*a_) * (a_*a_*tan_sq(com_arg(t, a_, b_, G_)) / (b_*b_) + 1) )
om(t) = om_th(t-t0, a, b, G)

#fit om(t) f u 1:(-$10) via a, b, t0, G

plot f u 1:(-$10) t "\omega_z" w p, f u 1:(om($1)) w l
