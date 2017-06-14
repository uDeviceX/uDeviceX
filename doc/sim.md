# sim.md

## conventions

Variables held by `sim` should be stored into abstract structures to facilitate readibility.
Wrapper functions take these abstract structures as arguments and dispatch the variables to finer functions, so one can see the input/output of these functions.

There are 3 kinds of abstract structures:

* [p] : states of the simulation variables such as `pp`, `np` etc.  
   They can be updated by other functions (ex: `[p] = update([p])`)
        
* [z] : Helper variables which should be made consistent with the
   above variables via some function.  ex: `zip` variables for
   solvent, used in dpd-forces computations: `ff = dpdforces([p],
   [z])`.  `sim` does not use them directly and should **NOT** modify
   them between calls or restore the state via `[w] = p2z([p])`
   function.
        
* [w] : Work variables. ex: exchange buffers in distribute
  functions.

## example

Let `u::` be a namespace with functions which follow the
conventions above.

sim::pp : solvent coordinates and velocity

sim::ww = upd::allocate_work()

sim::pp = sim::init_solvent()
sim::zz  = upd::p2z(sim::pp)

sim::ff = upd::get_force(sim::pp, sim::zz) # force  : a local for sim:: it does not have to follow conventions

[sim::pp, sim::zz] = upd::new_pos(sim::pp, sim::zz)   # have to pass and recive also zip

[sim::pp, sim::zz] = upd::distrib(sim::pp, sim::zz, sim::ww) # can be only one call
