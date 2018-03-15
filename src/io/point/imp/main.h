static char *cat(char *dest, const char *src) {
    return strncat(dest, src, FILENAME_MAX);
}

static char *cpy(char *dest, const char *src) {
    return strncpy(dest, src, FILENAME_MAX);
}

void io_point_conf_ini(/**/ IOPointConf **pq) {
    IOPointConf *q;
    EMALLOC(1, &q);
    q->i = 0;
    *pq = q;
}

void io_point_conf_push(IOPointConf *q, int nv, const char *k0) {
    int i;
    i = q->i;
    cpy(q->keys[i], k0);
    q->nn[i] = nv; q->i = i + 1;
}

void io_point_conf_fin(IOPointConf *q) { EFREE(q); }

void io_point_ini(int maxn, const char *path, IOPointConf *c, /**/ IOPoint **pq) {
    int i, n;
    IOPoint *q;
    char name[N_MAX*(FILENAME_MAX + 1)];
    EMALLOC(1, &q);
    n = c->i;
    for (i = 0; i < n; i++) {
        q->nn[i] = c->nn[i];
        cpy(q->keys[i], c->keys[i]);
    }

    for (i = 0; i < n; i++) {
        if (i > 0) cat(name, " ");
        cat(name, c->keys[i]);
    }
    
    cpy(q->path, path);
    q->n = n;
    q->maxn = maxn;
    *pq = q;
}

void io_point_fin(IOPoint *q) { EFREE(q); }
