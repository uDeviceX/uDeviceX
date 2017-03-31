

fhm = "< sed -n '30,200p' results/half_mass/solid_diag.txt"
fum = "< sed -n '30,200p' results/unit_mass/solid_diag.txt"

set xlabel "t * gammadot"
set ylabel "\omega_z / gammadot"

gammadot=0.05

set macro

rescale_om(om) = -om / gammadot
rescale_t (t ) =   t * gammadot

#omvst = 'u 1:10'
omvst = 'u (rescale_t($1)):(rescale_om($10))'

hm = 'fhm @omvst w p t "m = 0.5"'
um = 'fum @omvst w p t "m = 1"'

f(x) = a

fit f(x) fhm @omvst via a
ahm = a

fit f(x) fum @omvst via a
aum = a

plot @hm, @um, ahm, aum
