static void anti(Adj *adj, /**/ int *dev) {
    int n;
    int *hst;
    n = adj_get_max(adj);
    EMALLOC(n, &hst);
    adj_get_anti(adj, /**/ hst);
    cH2D(dev, hst, n);
    EFREE(hst);
}

static void setup0(Adj *adj, /**/ Shape *shape) {
    if (RBC_RND)         UC(anti(adj, /**/ shape->anti));
}

static void setup(int md, int nt, int nv, const int4 *tt, /**/ RbcQuants *q) {
    Adj *adj;
    UC(adj_ini(md, nt, nv, tt, /**/ &adj));
    UC(setup0(adj, /**/ &q->shape));
    UC(adj_fin(adj));
}
