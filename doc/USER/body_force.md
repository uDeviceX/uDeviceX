# body (external) force

Types of body forces is controled by `FORCE_*` parameters

* `FORCE_NONE` (default) no force
* `FORCE_DOUBLE_POISEUILLE`
  "double poiseuille" force in `X` with gradient in `Y`

* `FORCE_CONSTANT`
  constant force in `X`

* `FORCE_4ROLLER`
   see [src/scheme/force/dev/4roller.h](src/scheme/force/dev/4roller.h)
   
* `FORCE_SHEAR`
   shearing force in `X` and gradient in `Y`. Amplitude is
   `f = mass * FORCE_PAR_A * <distance from domain center in X>`

Amplitude for all forces except for `FORCE_SHEAR` is `FORCE_PAR_A`
