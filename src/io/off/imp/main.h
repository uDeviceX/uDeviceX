static void ini(OffRead **pq) {
    OffRead *p;
    UC(emalloc(sizeof(OffRead), (void**)&p));
    *pq = p;
}

static void read(FILE *f, const char *path, /**/ OffRead *q) {
    int nv, nt;
    /*
    header(f);
    comments(f);
    sizes(f, &nv, &nt);
    vert(f);
    tri(f); */

    //    q->nv = nv; q->nt = nt;
}
void off_read(const char *path, OffRead **pq) {
    FILE *f;
    OffRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read(f, path, /**/ q);
    UC(efclose(f));
    *pq = q;
}

void off_fin(OffRead* q) {
    //    UC(efree(q->rr));
    //    UC(efree(q->tt));
    UC(efree(q));
}

int    off_get_n(OffRead*) {
    return 0;
}

int4  *off_get_tri(OffRead*) {
    int4 *q = NULL;
    return q;
}

float *off_get_vert(OffRead*) {
    float *q = NULL;
    return q;
}
