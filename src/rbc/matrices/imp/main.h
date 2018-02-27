static const int MAX_N = 999999;

void matrices_read(const char*, /**/ Matrices **pq) {
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    *pq = q;
}

void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }
