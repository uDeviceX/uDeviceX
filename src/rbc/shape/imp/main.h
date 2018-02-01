void rbc_shape_ini(Adj *adj, const float *rr, /**/ RbcShape **pq) {
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

void rbc_shape_edg  (RbcShape *q, /**/ float** pe) { *pe = q->edg; }
void rbc_shape_area (RbcShape *q, /**/ float** pe) { *pe = q->area; }
void rbc_shape_anti (RbcShape *q, /**/ int**   pe) { *pe = q->anti; }

void rbc_shape_total_area(RbcShape *q, /**/ float *pe)   { *pe = q->A; }
void rbc_shape_total_volume(RbcShape *q, /**/ float *pe) { *pe = q->V; }
