void wall_fin_quants(WallQuants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void wall_fin_ticket(WallTicket *t) {
    UC(rnd_fin(t->rnd));
    UC(texo_destroy(&t->texstart));
    UC(texo_destroy(&t->texpp));
    UC(clist_fin(&t->cells));
    UC(efree(t));
}
