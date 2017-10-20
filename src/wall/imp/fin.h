void free_quants(Quants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void free_ticket(Ticket *t) {
    delete t->rnd;
    t->texstart.destroy();
    t->texpp.destroy();
    fin(&t->cells);
    fin_map(&t->mcells);
}
