void adj_fin(Adj *q) {
    EFREE(q->adj0);
    EFREE(q->adj1);
    EFREE(q);
}

void adj_view_fin(Adj_v *q) {
    Dfree(q->adj0);
    Dfree(q->adj1);
    EFREE(q);
}
