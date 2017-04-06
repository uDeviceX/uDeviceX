set macro

fname = "solid_diag.txt"

f = "< cat $1".fname
favg = "< sed -n '15,$p' $1".fname

set xlabel "t * gammadot"
set ylabel "\omega_z / gammadot"

gammadot=0.05

rescale_om(om) = -om / gammadot
rescale_t (t ) =   t * gammadot

omvst = 'u (rescale_t($1)):(rescale_om($10))'

mu = 0.5
mean(x) = mu

fit mean(x) favg @omvst via mu

set key right bottom
plot f @omvst w lp, 0.5, mu t sprintf("mu = %f", mu)
