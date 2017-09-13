# clists

Cell lists

## possible inputs:
* local array of particles `pplo`
* (optional) remote array of particles `ppre`

The arrays may have "holes" in it, in which case the particle is not kept in the cell lists

# output:
`starts`, `counts`, `pp`

# work:
* `ee` cell [e]ntries
* `ii` scattered indices
* `ws `workspace for scan
