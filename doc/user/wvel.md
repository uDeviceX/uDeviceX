# [W]all [vel]ocitie

Walls can have prescribe velocities. Time dependent velocity profile
is chosen by setting one of the variables:

* `WVEL_FLAT`

   `vx = (YS - ycenter) * WVEL_PAR_A`

* `WVEL_SIN`

   `vx = (YS - ycenter) * WVEL_PAR_A * sin(WVEL_PAR_W * t)`

* `WVEL_DUPIRE_UP`
* `WVEL_DUPIRE_DOWN`

The velocity is in `X` direction and shear direction is controlled by
`WVEL_PAR_Y` and `WVEL_PAR_Z`
