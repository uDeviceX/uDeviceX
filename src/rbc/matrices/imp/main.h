void matrices_read(const char*, /**/ Matrices **pq) {
    Matrices *q;
    EMALLOC(1, &q);
    *pq = q;
}

void matrices_fin(Matrices *q) { EFREE(q); }
