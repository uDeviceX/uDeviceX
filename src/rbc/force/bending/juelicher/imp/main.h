void juelicher_ini(const MeshRead *cell, Juelicher **pq) {
#   define  nxt(h)     he_nxt(he, (h))
#   define  flp(h)     he_flp(he, (h))
#   define  ver(h)     he_ver(he, (h))
#   define  hdg_edg(e) he_hdg_edg(he, (e))
    Juelicher *q;
    int md, nt, nv, ne, e, i, j, k, l, h, n, nn, nnf;
    const int4 *tt;
    HeRead *he_read;
    He *he;
    int4 *dih;
    
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    he_read_int4_ini(nv, nt, tt, /**/ &he_read);
    he_ini(he_read, /**/ &he);

    ne = he_ne(he);
    EMALLOC(ne, &dih);
    EMALLOC(ne, &q->dih);
    Dalloc(&q->dih, ne);
    
    for (e = 0; e < ne; e++) { /* i[jk]l */
        h = hdg_edg(e); n = nxt(h); nn = nxt(nxt(h));
        j = ver(h); k = ver(n); i = ver(nn);
        nnf = nxt(nxt(flp(h)));
        l = ver(nnf);
        dih[e].x = i; dih[e].y = j; dih[e].z = k; dih[e].w = l;
    }
    q->ne = ne;
    cH2D(q->dih, dih, ne);
        
    UC(adj_ini(md, nt, nv, tt, /**/ &q->adj));
    UC(adj_view_ini(q->adj, /**/ &q->adj_v));

    he_read_fin(he_read);
    he_fin(he);
    EFREE(dih);

    *pq = q;
}

void juelicher_fin(Juelicher *q) {
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    Dfree(q->dih);
    EFREE(q);
}
