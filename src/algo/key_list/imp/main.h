void KeyList_ini(KeyList **pq) {
    KeyList *q;
    EMALLOC(1, &q);
    q->nk = 0;
    *pq = q;
}
void KeyList_fin(KeyList *q) { EFREE(q); }

void KeyList_append(KeyList *q, const char *k) {
    int nk;
    nk = q->nk;
    if (nk == MAX_NK)
        ERR("nk=%d == MAX_NK=%d", nk, MAX_NK);
    cpy(q->keys[nk], k);
    q->ww[nk] = string_nword(k);
    q->mark[nk] = 0;
    q->nk = nk + 1;
}

int KeyList_has(KeyList *q, const char *k) {
    int i, nk;
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) return 1;
    return 0;
}

int KeyList_offset(KeyList *q, const char *k) {
    int i, nk, offset;
    nk = q->nk;
    for (i = offset = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return offset;
        offset += q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int KeyList_width(KeyList *q, const char *k) {
    int i, nk;
    nk = q->nk;
    for (i = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int KeyList_size(KeyList *q) {
    int i, nk, size;
    nk = q->nk;
    for (i = size = 0; i < nk; i++)
        size += q->ww[i];
    return size;
}

void KeyList_mark(KeyList *q, const char *k) {
    int i, nk;
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) {
            q->mark[i] = 1;
            return;
        }
    ERR("key `%s` is not found", k);
}

int  KeyList_marked(KeyList *q) {
    int i, nk;
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (q->mark[i] == 0) return 0;
    return 1;
}

void  KeyList_unmark(KeyList *q) {
    int i, nk;
    nk = q->nk;
    for (i = 0; i < nk; i++)
        q->mark[i] = 0;
}

void KeyList_log(KeyList *q) {
    int i, nk;
    nk = q->nk;    
    msg_print("<key_list");
    for (i = 0; i < nk; i++)
        msg_print("'%s' %d", q->keys[i], q->mark[i]);
    msg_print(">");
}
