# tcells

triangle cell lists

input: 
* faces (integers containing the indeces of the vertices of each triangle)
* vertices (Particles)

output:
* ids (array with indices of triangles), ordered such that ids belonging to same cell are stored consecutively  
  sizeof this array is not known in advance
* starts
* counts
