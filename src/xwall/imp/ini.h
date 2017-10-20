void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    t->rnd   = new rnd::KISS;
    ini(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM, /**/ &t->cells);
    ini_map(&t->cells, /**/ &t->mcells);
}

