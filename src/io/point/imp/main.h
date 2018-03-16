void io_point_conf_ini(/**/ IOPointConf **pq) {
    IOPointConf *q;
    EMALLOC(1, &q);
    q->i = 0;
    *pq = q;
}

static void push(IOPointConf *q, int nv, const char *k0) {
    int i;
    i = q->i;
    cpy(q->keys[i], k0);
    q->nn[i] = nv; q->i = i + 1;
}
void io_point_conf_push(IOPointConf *q, const char *key) {
    int nw;
    UC(nw = nword(key));
    msg_print("string, nword: '%s', %d", key, nw);
    push(q, nw, key);
}
void io_point_conf_fin(IOPointConf *q) { EFREE(q); }

static void ini_bop(int maxn, int n, const char *keys, BopData **pq) {
    BopData *q;
    bop_ini(&q);
    bop_set_n(maxn, q);
    bop_set_vars(n, keys, q);
    bop_set_type(BopDOUBLE, q);
    bop_alloc(q);

    *pq = q;
}
void io_point_ini(int maxn, const char *path, IOPointConf *c, /**/ IOPoint **pq) {
    int i, n, cum_n;
    IOPoint *q;
    char cum_key[N_MAX*(FILENAME_MAX + 1)];
    EMALLOC(1, &q);
    n = c->i;
    for (i = 0; i < n; i++)
        q->seen[i] = 0;

    for (i = 0; i < n; i++) {
        q->nn[i] = c->nn[i];
        cpy(q->keys[i], c->keys[i]);
    }

    for (i = 0; i < n; i++) {
        if (i > 0) cat(cum_key, " ");
        cat(cum_key, c->keys[i]);
    }

    cum_n = 0;
    for (i = 0; i < n; i++)
        cum_n += q->nn[i];
    ini_bop(maxn, cum_n, cum_key, &q->bop);

    UC(mkdir(DUMP_BASE, path));
    cpy(q->path, path);
    q->n = n;
    q->maxn = maxn;
    *pq = q;
}

void io_point_fin(IOPoint *q) {
    bop_fin(q->bop);
    EFREE(q);
}

void io_point_push(IOPoint *q, int ndata, double *D, const char *key) {
    int offset, n, i;
    if (ndata > q->maxn)
        ERR("ndata=%d > q->maxn=%d", ndata, q->maxn);

    n = q->n;
    offset = 0;
    for (i = 0; ; i++) {
        if (i == n) {UC(wrong_key(q, key)); ERR(""); }
        if (same_str(key, q->keys[i])) break;
        offset += q->nn[i];
    }
    if (q->seen[i]) ERR("key '%s' already seen", key);
    q->seen[i] = 1;
    
}

void io_point_write(IOPoint*, MPI_Comm, int) {
}
