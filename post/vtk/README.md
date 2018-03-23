# Convert udx output to legacy vtk

    u.vtk.face vtk ply bop
    u.vtk.edge vtk ply bop
    u.vtk.vert vtk ply bop

- a library to read ply subset used by udx
- a library-wrapper for libbop
- a library to write vtk

  face
  data: points, point data[n], tri data[n], edge data[n]
