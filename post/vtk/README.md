# Convert udx output to legacy vtk

	u.vtk.tri  vtk ply bop
	u.vtk.edge vtk ply bop
	u.vtk.vert vtk ply bop

- a library to read ply subset used by udx
- a library-wrapper for libbop
- a library to write vtk

data: points, point data[n], tri data[n], edge data[n]

	void vtk_conf_ini(MeshRead)
	void vtk_conf_tri(keys)
	void vtk_conf_fin()

	void vtk_ini(maxn, path, conf)
	void vtk_points(nm, Vectors)
	void vtk_tri(nm, data, keys)
	void vtk_write(MPI_Comm, id)
	void vtk_fin()
