# bulk particles
`pp` the array of all bulk particles  
`k = 0 .. n` loops over all bulk particles

# halo
`halo` is composed of 26 `frag` : fragments

# Fragments

Each fragment has `nc` cell infos:  
-start from bulk coordinates : `str`  
-counts : `cnt`  
-start from fragment coordinates : `cum`  

Each fragment also has (output of `pack`):  
-Particles `pp`  
-Particle indices `ii`  

# halo particles

`hid` is the fragment id `0 <= hid < 26`  
`pp[hid][i]` is the particle  

# bulk cell lists

`cid` is cell id  
`start[cid]` point to global particle id  
`pp[start[cid]]` is the first particle in the cell `cid`  

`count[cid]` is the number of particles in the cell `cid`  

`k = start[cid] ... start[cid] + count[cid]` loops over all particles in the cell `cid`  

# halo cell lists

`hid` is fragment id : `0 <= hid < 26`  
`hci` is fragment cell id : `0 < hci < nc[hid]`  

`hstart[hid][hci]`  is the first particle in the corresponding bulk cell  
`hcount[hid][hci]`  is the number of particles in the corresponding bulk cell  

`k = hstart[hid][hci] ... hstart[hid][hci] + hcount[hid][hci]` loops over all particles in the corresponding bulk cell   

`k = hcumul[hid][hci] ... hcumul[hid][hci] + hcount[hid][hci]` loops
over all the particles in the cell `hci` of the fragment particles `pp[hid]`
