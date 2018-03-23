# Convert udx output to legacy vtk

	u.vtk.face vtk ply bop
	u.vtk.edge vtk ply bop
	u.vtk.vert vtk ply bop

- a library to read ply subset used by udx
- a library-wrapper for libbop
- a library to write vtk

data: points, point data[n], tri data[n], edge data[n]

	conf_ini(MeshRead)
	conf_push_edge(keys)
	conf_fin()

	ini(maxn, path, conf)
	push_points(n, Vectors)
	push_edge(n, data, keys)
	write(id)
	fin()
