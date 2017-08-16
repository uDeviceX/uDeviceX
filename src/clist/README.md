# cell list structure for particles

contains two integer vectors of size ncells.  the start[cell-id] array
gives the entry in the particle array associated to first particle
belonging to cell-id count[cell-id] tells how many particles are
inside cell-id.  Building the cell lists reorders the particle array.
