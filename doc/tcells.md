# tcells

triangle cell lists

## structure

see `Quants` structure:

* ids (array with indices of triangles), ordered such that ids belonging to same cell are stored consecutively  
  sizeof this array is not known in advance
* starts
* counts

## build
input: 

* faces (integers containing the indeces of the vertices of each triangle)
* vertices (Particles)

output: 
* tcells quants
