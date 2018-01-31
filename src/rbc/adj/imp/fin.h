void adj_fin(Adj* A) {
    EFREE(A->adj0);
    EFREE(A->adj1);
}
void adj_view_fin(Adj_v *q) { EFREE(q); }
