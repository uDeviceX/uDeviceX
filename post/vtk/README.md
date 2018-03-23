# Convert udx output to legacy vtk

	u.vtk.face vtk ply bop
	u.vtk.edge vtk ply bop
	u.vtk.vert vtk ply bop

- a library to read ply subset used by udx
- a library-wrapper for libbop
- a library to write vtk

data: points, point data[n], tri data[n], edge data[n]

	void KeyList_ini()
	void KeyList_copy(KeyList*, /**/ KeyList**)
	void KeyList_append(key)
	bool KeyList_has(key)
	int KeyList_offset(key)
	int KeyList_width(i)
	int KeyList_size()

	void KeyList_mark(key)
	void KeyList_clear()
	void KeyList_marked()

	void KeyList_fin()

	conf_ini(MeshRead)
	conf_push_edge(keys)
	conf_fin()

	ini(maxn, path, conf)
	push_points(n, Vectors)
	push_edge(n, data, keys)
	write(id)
	fin()
