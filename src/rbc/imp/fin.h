void fin(Quants *q) {
    Dfree(q->pp);
    Dfree(q->av);

    Dfree(q->tri);
    Dfree(q->adj0);
    Dfree(q->adj1);

    delete[] q->tri_hst;
    delete[] q->pp_hst;
}

void fin_ticket(TicketT *t) {
    t->textri.destroy();
    t->texadj0.destroy();
    t->texadj1.destroy();
    t->texvert.destroy();
}
