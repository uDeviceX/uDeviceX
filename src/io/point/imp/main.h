#define PATTERN "%s/%05d"

void io_point_conf_ini(/**/ IOPointConf **pq) {
    IOPointConf *q;
    EMALLOC(1, &q);
    q->i = 0;
    *pq = q;
}

static void conf_push(IOPointConf *q, int nv, const char *k0) {
    int i;
    i = q->i;
    cpy(q->keys[i], k0);
    q->nn[i] = nv; q->i = i + 1;
}
void io_point_conf_push(IOPointConf *q, const char *key) {
    int nw;
    UC(nw = nword(key));
    msg_print("string, nword: '%s', %d", key, nw);
    conf_push(q, nw, key);
}
void io_point_conf_fin(IOPointConf *q) { EFREE(q); }

static void ini_bop(int maxn, int n, const char *keys, BopData **pq) {
    BopData *q;
    BPC(bop_ini(&q));
    BPC(bop_set_n(maxn, q));
    BPC(bop_set_vars(n, keys, q));
    BPC(bop_set_type(BopDOUBLE, q));
    BPC(bop_alloc(q));

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

    cum_n = 0;
    for (i = 0; i < n; i++) {
        q->nn[i] = c->nn[i];
        cum_n += q->nn[i];
        cpy(q->keys[i], c->keys[i]);
    }

    for (i = 0; i < n; i++) {
        if (i > 0) cat(cum_key, " ");
        cat(cum_key, c->keys[i]);
    }

    ini_bop(maxn, cum_n, cum_key, &q->bop);

    UC(mkdir(DUMP_BASE, path));
    cpy(q->path, path);
    q->n = n;
    q->maxn = maxn;
    q->cum_n = cum_n;
    *pq = q;
}

void io_point_fin(IOPoint *q) {
    BPC(bop_fin(q->bop));
    EFREE(q);
}

static double *get_data(BopData *b) {
    BopType type;
    BPC(bop_get_type(b, &type));
    if (type != BopDOUBLE) ERR("BopType is not double");
    return (double*)bop_get_data(b);
}

static void push(int n, int nvar, int cum_n, const double *F, int offset, double *T) {
    int i, j, f, t; /* from/to */
    for (f = i = 0; i < n; i++) {
        for (j = 0; j < nvar; j++) {
            t = cum_n * i + offset;
            T[t] = F[f++];
        }
    }
}
void io_point_push(IOPoint *q, int ndata, const double *D, const char *key) {
    int offset, n, i;
    double *B;
    if (ndata > q->maxn)
        ERR("ndata=%d > q->maxn=%d; key: '%s'",
            ndata, q->maxn, key);

    n = q->n;
    offset = 0;
    for (i = 0; ; i++) {
        if (i == n) {UC(wrong_key(q, key)); ERR(""); }
        if (same_str(key, q->keys[i])) break;
        offset += q->nn[i];
    }
    if (q->seen[i]) ERR("seen key '%s' ", key);
    q->seen[i] = 1;
    UC(B = get_data(q->bop));
    push(ndata, q->nn[i], q->cum_n, D, offset, /**/ B);
}

void io_point_write(IOPoint *q, MPI_Comm comm, int id) {
    char name[FILENAME_MAX];
    int i, n;
    n = q->n;

    for (i = 0; i < n; i++)
        if (!q->seen[i])
            ERR("key '%s' was not pushed", q->keys[i]);

    if (snprintf(name, FILENAME_MAX, PATTERN, q->path, id) < 0)
        ERR("snprintf failed");

    BPC(bop_summary(q->bop));
    //    UC(set_rank_infos(cart, n, q->bop));
    BPC(bop_write_header(comm, name, q->bop));
    BPC(bop_write_values(comm, name, q->bop));

    for (i = 0; i < n; i++)
        q->seen[i] = 0;
}
