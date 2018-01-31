void wall_ini_quants(int3 L, WallQuants *q) {
    q->n = 0;
    q->pp = NULL;
    q->L = L;
}

void wall_ini_ticket(WallTicket **ti) {
    WallTicket *t;
    EMALLOC(1, ti);
    t = *ti;
    UC(rnd_ini(42, 42, 42, 42, /**/ &t->rnd));
    clist_ini(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM, /**/ &t->cells);
}
