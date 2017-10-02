# exch/mesh

## basic
mesh exchanger: exchange overlapping meshes accross nodes  
used for coloring solvent  

## options
* send back total linear and angular momentum of each rig object (rig bounce-back)
* send back linear and angular momentum for each triangle (rbc bounce-back)

## compressed Momentum arrays `mm`

most triangle do not collide with pp at every timestep -> compress `mm`

* `mmc` non zero entries of `mm`
* `ii` triangle ids (code that also contains the mesh id)

