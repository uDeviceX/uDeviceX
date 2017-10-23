void free_quants(Quants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void free_ticket(Ticket *t) {
    delete t->rnd;
    destroy(&t->texstart);
    destroy(&t->texpp);
    fin(&t->cells);
    fin_map(&t->mcells);
}
