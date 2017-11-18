static void dealloc(AdjHst* A) {
    free(A->adj0);
    free(A->adj1);
}
void fin(AdjHst* A) { dealloc(A); }
