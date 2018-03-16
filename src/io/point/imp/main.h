static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }

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
static int nword(const char *s) {
    enum {OUT, IN};
    char c;
    int state, n;
    state = OUT;
    n = 0;
    while ((c = *s++) != '\0') {
        if (!(isalnum(c) || isspace(c)))
            ERR("illegal character '%c' in '%s'", c, s);
        else if (isspace(c))
            state = OUT;
        else if (state == OUT) {
            state = IN;
            n++;
        }
    }
    if (n == 0) ERR("wrong keys for bop: '%s'", s);
    return n;
}
void io_point_conf_push(IOPointConf *q, const char *key) {
    int nw;
    UC(nw = nword(key));
    msg_print("s, nw: '%s', %d", key, nw);
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
    UC(os_mkdir(DUMP_BASE "/com"));
    cpy(q->path, path);
    q->n = n;
    q->maxn = maxn;
    *pq = q;
}

void io_point_fin(IOPoint *q) {
    bop_fin(q->bop);
    EFREE(q);
}
