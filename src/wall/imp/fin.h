void wall_fin_quants(WallQuants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void wall_fin_ticket(Ticket *t) {
    UC(rnd_fin(t->rnd));
    destroy(&t->texstart);
    destroy(&t->texpp);
    clist_fin(&t->cells);
}
