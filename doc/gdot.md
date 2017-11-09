# "Gamma dot"

Walls can have prescribe velocities. Time dependent velocity profile
is chosen by setting one of the variables:

* `GDOT_FLAT`

   `vy = (XS - xcenter) * GDOT_PAR_A`

* `GDOT_SIN`

   `vy = (XS - xcenter) * GDOT_PAR_A * sin(GDOT_PAR_W * t)`

* `GDOT_DUPIRE_UP`
* `GDOT_DUPIRE_DOWN`
