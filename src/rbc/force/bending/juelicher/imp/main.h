void juelicher_ini(const MeshRead *cell, Juelicher **pq) {
#   define  nxt(h)     he_nxt(he, (h))
#   define  flp(h)     he_flp(he, (h))
#   define  ver(h)     he_ver(he, (h))
#   define  hdg_edg(e) he_hdg_edg(he, (e))
    Juelicher *q;
    int nt, nv, ne, e, i, j, k, l, h, n, nn, nnf;
    const int4 *tri;
    HeRead *he_read;
    He *he;
    int4 *dih;

    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tri = mesh_read_get_tri(cell);

    he_read_int4_ini(nv, nt, tri, /**/ &he_read);
    he_ini(he_read, /**/ &he);

    ne = he_ne(he);
    EMALLOC(ne, &dih);
    Dalloc(&q->dih, ne);
    Dalloc(&q->tri, nt);

    Dalloc(&q->theta,    MAX_CELL_NUM*ne); /* (sic) */
    Dalloc(&q->area,     MAX_CELL_NUM*nv);
    Dalloc(&q->lentheta, MAX_CELL_NUM*nv);

    Dalloc(&q->lentheta_tot, MAX_CELL_NUM);
    Dalloc(&q->area_tot,     MAX_CELL_NUM);
    Dalloc(&q->curva_mean_area_tot,     MAX_CELL_NUM);
    Dalloc(&q->f,   3*MAX_CELL_NUM*nv);
    Dalloc(&q->fad, 3*MAX_CELL_NUM*nv);    

    for (e = 0; e < ne; e++) { /* i[jk]l */
        h = hdg_edg(e); n = nxt(h); nn = nxt(nxt(h));
        j = ver(h); k = ver(n); i = ver(nn);
        nnf = nxt(nxt(flp(h)));
        l = ver(nnf);
        dih[e].x = i; dih[e].y = j; dih[e].z = k; dih[e].w = l;
    }

    q->ne = ne;
    cH2D(q->dih, dih, ne);
    cH2D(q->tri, tri,  nt);

    he_read_fin(he_read);
    he_fin(he);
    EFREE(dih);

    *pq = q;
}

void juelicher_fin(Juelicher *q) {
    Dfree(q->dih);
    Dfree(q->tri);
    Dfree(q->area);
    Dfree(q->theta);
    Dfree(q->lentheta);
    Dfree(q->lentheta_tot);
    Dfree(q->area_tot);
    Dfree(q->curva_mean_area_tot);
    Dfree(q->f);
    Dfree(q->fad);
    EFREE(q);
}
