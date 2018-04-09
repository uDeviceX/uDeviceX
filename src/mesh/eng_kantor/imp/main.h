void mesh_eng_kantor_ini(MeshRead *mesh, int nm, /**/ MeshEngKantor **pq) {
    int nv, ne;
    MeshEngKantor *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->ne = ne;
    EMALLOC(nm*ne, &q->angles);
    mesh_angle_ini(mesh, /**/ &q->angle);

    *pq = q;
}

void mesh_eng_kantor_fin(MeshEngKantor *q) {
    EFREE(q->angles);
    mesh_angle_fin(q->angle);
    EFREE(q);
}

void mesh_eng_kantor_apply(MeshEngKantor *q, int nm, Vectors *pos, double kb, double angle0, /**/ double *o) {
    int i, ne;
    double *angles, angle;
    angles = q->angles;
    ne = q->ne;
    mesh_angle_apply(q->angle, nm, pos, /**/ angles);
    for (i = 0; i < nm * ne; i++) {
        angle = angles[i];
        o[i] = kb * (1 - cos(angle - angle0));
    }
}
