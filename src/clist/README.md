# clists

Cell lists  
Purpose: reorder data according to particle positions

## possible inputs:
* local array of particles `pplo`, size `nlo`
* (optional) remote array of particles `ppre`, size `nre`

The arrays may have "holes" in it, in which case the missing particle is not kept in the cell lists.
in general, `nlo` is bigger than the number of local particles because of these holes.

## output:
`starts`, `counts`, `pp`
`pp` has size `nout`  
`nout` is __not__ equal to nlo + nre in general!

## ticket:
* `ee` cell [e]ntries (`lo` and `re` versions)
* `ii` scattered indices
* `ws `workspace for scan

