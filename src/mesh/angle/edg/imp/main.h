void e_ini(int md, int nv, /**/ Edg **pq) {
    Edg *q;
    EMALLOC(1, &q);
    EMALLOC(md*nv, &q->hx);
    EMALLOC(md*nv, &q->hy);
    
    q->md = md; q->nv = nv;
    *pq = q;
}

void e_fin(Edg *q) {
    EFREE(q->hx);
    EFREE(q->hy);
    EFREE(q);
}
