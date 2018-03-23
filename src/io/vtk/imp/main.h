void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    EMALLOC(1, &q);
}

void vtk_conf_fin(VTKConf *q) { EFREE(q); }
