set grid

dt=5e-4
fd=100
Dt=fd*dt

set xlabel "t"
set ylabel "MSD"

set macros

time = 'Dt*$0'
data = '"tmp" u (@time):1'

linear(x) = a*x + b
fit linear(x) @data via a, b

plot @data w p title "", "tmp" u (@time):(a*@time + b) w l title sprintf("%f * x + %f", a, b)
