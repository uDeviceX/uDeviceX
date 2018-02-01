void rbc_shape_ini(Adj *adj, float *rr, /**/ RbcShape **pq) {
    int n;
    RbcShape *q;
    n = adj_get_max(adj);
    
    EMALLOC(1, &q);
    EMALLOC(n, &q->edg);
    EMALLOC(n, &q->area);
    EMALLOC(n, &q->anti);

    *pq = q;
}

void rbc_shape_fin(RbcShape *q) {
    EFREE(q->edg);
    EFREE(q->area);
    EFREE(q->anti);    
    EFREE(q);
}
