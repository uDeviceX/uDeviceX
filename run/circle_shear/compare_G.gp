

fg1 = "< sed -n '30,200p' solid_diag.txt"
fg2 = "< sed -n '30,200p' results/unit_mass/solid_diag.txt"

set xlabel "t * gammadot"
set ylabel "\omega_z / gammadot"

gammadot=0.05

set macro

rescale_om(om) = -om / gammadot
rescale_t (t ) =   t * gammadot

#omvst = 'u 1:10'
omvst = 'u (rescale_t($1)):(rescale_om($10))'

g1 = 'fg1 @omvst w p t "G = 0.025"'
g2 = 'fg2 @omvst w p t "G = 0.050"'

f(x) = a

fit f(x) fg1 @omvst via a
a1 = a

fit f(x) fg2 @omvst via a
a2 = a

plot @g1, @g2, a1, a2
