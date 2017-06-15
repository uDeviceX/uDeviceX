# sim.md

## conventions

Variables held by `sim` should be stored into abstract. Wrapper
functions take these abstract structures as arguments and dispatch the
variables to finer functions, so one can see the input/output of these
functions.

There are 3 kinds of abstract structures:

* Q : quntaties : states of the simulation variables such as `pp`, `np` etc.
   They can be updated by other functions (ex: `[p] = update([p])`)

* W : work : Work variables. ex: exchange buffers in distribute
  functions.

* T : tickets : Helper variables which should be made consistent with
   the above variables via some function.  ex: `zip` variables for
   solvent, used in dpd-forces computations: `ff = dpdforces([p],
   [z])`.  `sim` does not use them directly and should **NOT** modify
   them between calls
