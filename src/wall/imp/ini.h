void alloc_quants(WallQuants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    UC(rnd_ini(42, 42, 42, 42, /**/ &t->rnd));
    clist_ini(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM, /**/ &t->cells);
}
