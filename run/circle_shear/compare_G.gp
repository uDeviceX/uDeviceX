
fg1 = "< sed -n '30,200p' solid_diag.txt"
fg2 = "< sed -n '30,200p' results/unit_mass/solid_diag.txt"

set xlabel "t * gammadot"
set ylabel "\omega_z / gammadot"

gammadot1=0.025
gammadot2=0.050

set macro

rescale_om(om, G) = -om / G
rescale_t (t , G) =   t * G

g1 = 'fg1 u (rescale_t($1, gammadot1)):(rescale_om($10, gammadot1)) w p t "G = 0.025"'
g2 = 'fg2 u (rescale_t($1, gammadot2)):(rescale_om($10, gammadot2)) w p t "G = 0.050"'

f(x) = a

fit f(x) fg1 u (rescale_t($1, gammadot1)):(rescale_om($10, gammadot1)) via a
a1 = a

fit f(x) fg2 u (rescale_t($1, gammadot2)):(rescale_om($10, gammadot2)) via a
a2 = a

plot @g1, @g2, a1, a2
