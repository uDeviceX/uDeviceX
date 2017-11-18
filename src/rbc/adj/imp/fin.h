static void dealloc(Hst* A) {
    free(A->adj0);
    free(A->adj1);
}
void fin(Hst* A) { dealloc(A); }
