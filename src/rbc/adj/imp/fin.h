static void dealloc(Adj* A) {
    free(A->adj0);
    free(A->adj1);
}
void fin(Adj* A) { dealloc(A); }
