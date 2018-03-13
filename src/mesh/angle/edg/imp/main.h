void e_ini(int md, int nv, int val0, /**/ Edg **pq) {
    int i;
    Edg *q;
    EMALLOC(1, &q);
    EMALLOC(md*nv, &q->hx);
    EMALLOC(md*nv, &q->hy);
    edg_ini(md, nv, /**/ q->hx);
    for (i = 0; i < md*nv; i++)
        q->hy[i] = val0;
    
    q->md = md; q->nv = nv;
    *pq = q;
}

void e_fin(Edg *q) {
    EFREE(q->hx);
    EFREE(q->hy);
    EFREE(q);
}

void e_set(Edg *q, int i, int j, int val) {
    int md, *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    UC(edg_set(md, i, j, val, hx, hy));
}

int e_get(Edg *q, int i, int j) {
    int md;
    const int *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    return edg_get(md, i, j, hx, hy);
}
