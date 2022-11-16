void key_list_ini(KeyList **pq) {
    KeyList *q;
    EMALLOC(1, &q);
    q->nk = 0;
    q->stamp = STAMP_GOOD;
    *pq = q;
}

void key_list_copy(KeyList *a, /**/ KeyList **pq) {
    int i, nk;
    KeyList *q;
    EMALLOC(1, &q);
    nk = a->nk;
    for (i = 0; i < nk; i++) {
        q->ww[i] = a->ww[i];
        q->mark[i] = a->mark[i];
        cpy(q->keys[i], a->keys[i]);
    }
    q->stamp = STAMP_GOOD;
    q->nk = nk;
    *pq = q;
}

void key_list_fin(KeyList *q) {
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    EFREE(q);
}

void key_list_push(KeyList *q, const char *k) {
    int nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    if (nk == MAX_NK)
        ERR("nk=%d == MAX_NK=%d", nk, MAX_NK);
    cpy(q->keys[nk], k);
    q->ww[nk] = string_nword(k);
    q->mark[nk] = 0;
    q->nk = nk + 1;
}

int key_list_has(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) return 1;
    return 0;
}

int key_list_offset(KeyList *q, const char *k) {
    int i, nk, offset;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = offset = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return offset;
        offset += q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_width(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_size(KeyList *q) {
    int i, nk, size;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = size = 0; i < nk; i++)
        size += q->ww[i];
    return size;
}

const char* key_list_name(KeyList *q, int i) {
    int n;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    n = key_list_size(q);
    if (i < 0) ERR("i=%d < 0", i);
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    return q->keys[i];
}

void key_list_mark(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) {
            q->mark[i] = 1;
            return;
        }
    ERR("key `%s` is not found", k);
}

int  key_list_marked(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (q->mark[i] == 0) return 0;
    return 1;
}

void  key_list_unmark(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        q->mark[i] = 0;
}

void key_list_log(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    msg_print("<key_list");
    for (i = 0; i < nk; i++)
        msg_print("'%s' %d", q->keys[i], q->mark[i]);
    msg_print(">");
}
