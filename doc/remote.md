# halo
`halo` is composed of 26 `frag` : fragments


# bulk particles
`pp` the array of all bulk particles
`pp[i]` is the particle
`k = 0 .. n` loops over all bulk particles

# halo particles
`hid` is halo id `0 < hid < 26` (!)
`pp[hid][i]` is the particle

# bulk cell lists
`cid` is cell id
`start[cid]` point to global particle id
`pp[start[cid]]` is the first particle in the cell `cid`

`count[cid]` is the number of particles in the cell `cid`

`k = start[cid] ... start[cid] + count[cid]` loops over all particles in the cell `cid`

# halo cell lists
`hid` is halo id : `0 < hid < 26` (!)
`hci` is halo cell id : `0 < hci < [number of cells for a halo hid]`

`hstart[hid][hci]`  is the first particle in one bulk celll
`hcount[hid][hci]`  is the number of particles in one bulk cell

`k = hstart[hid][hci] ... hstart[hid][hci] + hcount[hid][hci]` loops over all particles in one bulk cell 

`k = hcumul[hid][hci] ... hcumul[hid][hci] + hcount[hid][hci]` loops
over all the particles in the cell `hci` of `pp[hid]`
