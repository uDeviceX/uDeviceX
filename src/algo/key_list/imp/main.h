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
    q->mark[nk] = 0;
    q->nk = nk + 1;
}
