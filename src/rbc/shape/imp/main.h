void compute_edg(Adj *adj, const float *rr, /**/ float *edg) {
    int i, valid, n;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        msg_print("ijk: %d %d %d", m.i0, m.i1, m.i2);
    }
}

void rbc_shape_ini(Adj *adj, const float *rr, /**/ RbcShape **pq) {
    int n;
    RbcShape *q;
    n = adj_get_max(adj);
    EMALLOC(1, &q);
    EMALLOC(n, &q->edg);
    EMALLOC(n, &q->area);
    EMALLOC(n, &q->anti);

    compute_edg(adj, rr, /**/ q->edg);

    *pq = q;
}

void rbc_shape_fin(RbcShape *q) {
    EFREE(q->edg);
    EFREE(q->area);
    EFREE(q->anti);
    EFREE(q);
}

void rbc_shape_edg  (RbcShape *q, /**/ float** pe) { *pe = q->edg; }
void rbc_shape_area (RbcShape *q, /**/ float** pe) { *pe = q->area; }
void rbc_shape_anti (RbcShape *q, /**/ int**   pe) { *pe = q->anti; }

void rbc_shape_total_area(RbcShape *q, /**/ float *pe)   { *pe = q->A; }
void rbc_shape_total_volume(RbcShape *q, /**/ float *pe) { *pe = q->V; }
