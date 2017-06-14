# sim.md

## conventions

There are 3 kinds of variables:

* [p] : states of the simulation variables such as `pp`, `np` etc.
        They can be updated by other functions (ex: `[p] = update([p])`)
        
* [z] : Helper variables which should be made consistent with the above variables via some `p2z` function.
        ex: `zip` variables for solvent, used in dpd-forces computations
        
* [w] : Work variables, sim do not use them directly and should NOT modify them between 2 calls
        ex: exchange buffers in distribute functions

