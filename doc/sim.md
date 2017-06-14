# sim.md

## conventions

Variables held by `sim` should be stored into abstract structures to facilitate readibility.
Wrapper functions take these abstract structures as input and dispatch the variables to finer functions, so one can see the input/output of thes functions.

There are 3 kinds of abstract structures:

* [p] : states of the simulation variables such as `pp`, `np` etc.  
   They can be updated by other functions (ex: `[p] = update([p])`)
        
* [z] : Helper variables which should be made consistent with the above variables via some `p2z` function.  
   ex: `zip` variables for solvent, used in dpd-forces computations: `ff = dpdforces([z])`
        
* [w] : Work variables, `sim` does not use them directly and should **NOT** modify them between 2 calls.  
   ex: exchange buffers in distribute functions

