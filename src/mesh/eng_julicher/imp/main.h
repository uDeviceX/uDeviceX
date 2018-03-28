void mesh_eng_julicher_ini(MeshRead *mesh, int nm, /**/ MeshEngJulicher **pq) {
    int nv, nt, ne;
    MeshEngJulicher *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(nt = mesh_read_get_nt(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->nt = nt; q->ne = ne;
    EMALLOC(nm*ne, &q->lens);
    EMALLOC(nm*ne, &q->angles);
    EMALLOC(nm*nv, &q->areas);    
    *pq = q;
}

void mesh_eng_julicher_fin(MeshEngJulicher *q) { EFREE(q); }
void mesh_eng_julicher_apply(MeshEngJulicher *q, int nm, Vectors *pos, double *o) {

}
