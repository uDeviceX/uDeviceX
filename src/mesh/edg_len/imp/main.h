void mesh_edg_len_ini(MeshRead *mesh, MeshEdgLen **pq) {
    MeshEdgLen *q;
    int i, j, nv, ne;
    int *ee;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y; ee[j++] = d.z;
    }
    q->nv = nv; q->ne = ne; q->ee = ee;
    *pq = q;
}

static void edg2vert(MeshEdgLen *q, Scalars *sc, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = scalars_get(sc, i + offset_e));
        o[a + offset_v] += value;
        o[b + offset_v] += value;
    }
}
void mesh_edg_len_apply(MeshEdgLen *q, int nm, Scalars *sc, /**/ double *o) {
    int i, j, k, d, n, nv, ne;
    int offset_v, offset_e;
    nv = q->nv;
    ne = q->ne;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(edg2vert(q, sc, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
    for (i = j = 0; i < nm; i++)
        for (k = 0; k < n; k++) {
            d = q->deg[k];
            if (d <= 0) ERR("wrong vert. degree: %d", d);
            if (j >= n) ERR("j=%d >= n=%d", j, n);
            o[j++] /= d;
        }
}

void mesh_edg_len_fin(MeshEdgLen *q) {
    EFREE(q->ee);
    EFREE(q->deg);
    EFREE(q);
}
