void ini0(D* d, int n) {
    Dalloc(&d->r, n);
}
void ini(D** pd, int n) {
    D* d;
    d = (D*)malloc(sizeof(D));
    ini0(d, n);
    *pd = d;
}

void fin0(D* d) {
    Dfree(d->r);
}
void fin(D* d) {
    fin0(d);
    free(d);
}
void nxt(D*) { }
void get_hst(const D*, int i) { }
