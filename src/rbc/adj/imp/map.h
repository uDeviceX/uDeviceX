static int fetch(int *adj, int i) { return adj[i]; }
static int hst0(int md, int nv, int i, int *adj0, int *adj1, /**/ Map *m) {
    int i0, i1, i2, i3, i4;
    int j, k;
    assert(i < md * nv);

    i0 = i / md;
    j  = i % md;

    k   = i0 * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0; /* invalid */

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}

int hst(int md, int nv, int i, Adj *A, /**/ Map *m) {
    int *adj0, *adj1;
    adj0 = A->adj0;
    adj1 = A->adj1;
    return hst0(md, nv, i, adj0, adj1, m);
}
