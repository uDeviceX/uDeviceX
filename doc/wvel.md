# [W]all [vel]ocitie

Walls can have prescribe velocities. Time dependent velocity profile
is chosen by setting one of the variables:

* `WVEL_FLAT`

   `vy = (XS - xcenter) * WVEL_PAR_A`

* `WVEL_SIN`

   `vy = (XS - xcenter) * WVEL_PAR_A * sin(WVEL_PAR_W * t)`

* `WVEL_DUPIRE_UP`
* `WVEL_DUPIRE_DOWN`
