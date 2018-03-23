void KeyList_ini(KeyList **pq) {
    KeyList *q;
    EMALLOC(1, &q);
    *pq = q;
}
void KeyList_fin(KeyList *q) { EFREE(q); }
