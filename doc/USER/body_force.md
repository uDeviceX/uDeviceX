# body (external) force

Types of body forces is controled by `FORCE_*` parameters

* `FORCE_NONE` (default) no force
* `FORCE_DOUBLE_POISEUILLE`
  "double poiseuille" force in `X' direction with gradient in `Y'

* `FORCE_CONSTANT`
  constant force in `X` direction

* `FORCE_4ROLLER`
   see [src/scheme/dev/force/4roller.h](src/scheme/dev/force/4roller.h)

Amplitude for all forces is `FORCE_PAR_A`
