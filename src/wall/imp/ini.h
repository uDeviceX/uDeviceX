void wall_ini_quants(int3 L, WallQuants *q) {
    q->n = 0;
    q->pp = NULL;
    q->L = L;
}

void wall_ini_ticket(int3 L, WallTicket **ti) {
    WallTicket *t;
    EMALLOC(1, ti);
    t = *ti;
    UC(rnd_ini(42, 42, 42, 42, /**/ &t->rnd));
    UC(clist_ini(L.x + 2 * XWM,
                 L.y + 2 * YWM,
                 L.z + 2 * ZWM, /**/ &t->cells));
}
