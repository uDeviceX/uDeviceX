f = "< cat solid_diag.txt"
favg = "< sed -n '15,$p' solid_diag.txt"

set xlabel "t * gammadot"
set ylabel "\omega_z / gammadot"

gammadot=0.05

set macro

rescale_om(om) = -om / gammadot
rescale_t (t ) =   t * gammadot

omvst = 'u (rescale_t($1)):(rescale_om($10))'

mu = 0.5
mean(x) = mu

fit mean(x) favg @omvst via mu

set key right bottom
plot f @omvst w lp, 0.5, mu t sprintf("mu = %f", mu)
