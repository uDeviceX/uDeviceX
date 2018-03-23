void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    int nt, nv, i;
    const int4 *tt;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    tt = mesh_read_get_tri(mesh);
    EMALLOC(nt, &q->mesh->tt);
    for (i = 0; i < nt; i++)
        q->mesh->tt[i] = tt[i];
    q->mesh->nv = nv;
    q->mesh->nt = nt;

    key_list_ini(&q->tri);
    *pq = q;
}

void vtk_conf_fin(VTKConf *q) {
    key_list_fin(q->tri);
    EFREE(q->mesh->tt);
    EFREE(q);
}

void vtk_conf_tri(VTKConf *q, const char *keys) {
    UC(key_list_push(q->tri, keys));
}
