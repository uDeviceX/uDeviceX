void free_quants(Quants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void free_ticket(Ticket *t) {
    UC(rnd_fin(t->rnd));
    destroy(&t->texstart);
    destroy(&t->texpp);
    clist_fin(&t->cells);
}
