enum {
    IN,
    OUT
};

static void exchange_mesh(int maxm, int3 L, MPI_Comm cart, int nv, /*io*/ int *n, Particle *pp) {
    EMeshPack *pack;
    EMeshComm *comm;
    EMeshUnpack *unpack;
    int nm, nmhalo;
    nm = *n / nv;

    UC(emesh_pack_ini(L, nv, maxm, &pack));
    UC(emesh_comm_ini(cart, /**/ &comm));
    UC(emesh_unpack_ini(L, nv, maxm, &unpack));

    UC(emesh_build_map(nm, nv, pp, /**/ pack));
    UC(emesh_pack(nv, pp, /**/ pack));
    UC(emesh_download(pack));

    UC(emesh_post_recv(comm, unpack));
    UC(emesh_post_send(pack, comm));
    UC(emesh_wait_recv(comm, unpack));
    UC(emesh_wait_send(comm));

    UC(emesh_unpack(nv, unpack, /**/ &nmhalo, pp + nm * nv));

    *n += nm * nv;
    
    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

static void compute_labels(int pdir, int n, const Particle *pp, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh, /**/ int *ll) {
    float3 *lo, *hi;
    Triangles tri;
    Dalloc(&lo, nm);
    Dalloc(&hi, nm);

    tri.nt = nt;
    tri.tt = (int4*) tt;
    
    UC(minmax(pp_mesh, nv, nm, /**/ lo, hi));
    UC(collision_label(pdir, n, pp, &tri, nv, nm, pp_mesh, lo, hi, IN, OUT, /**/ ll));    
    Dfree(lo);
    Dfree(hi);
}

static void label_template_dev(int pdir, int3 L, MPI_Comm cart, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh,
                               int n_flu, const Particle *pp_flu, /**/ int *ll) {
    int maxm, n;
    Particle *pp;

    maxm = NFRAGS;
    Dalloc(&pp, nv * maxm);

    n = nm * nv;
    if (n) cD2D(pp, pp_mesh, n);
    
    UC(exchange_mesh(maxm, L, cart, nv, /* io */ &n, pp));
    UC(compute_labels(pdir, n_flu, pp_flu, nt, nv, nm, tt, pp_mesh, ll));

    Dfree(pp);
}

static void extract_in_hst(int n, const Particle *pp, const int *labels, int *nx, float *rrx) {
    enum {X, Y, Z};
    int i, j;
    const float *r;
    float *rx;
    for (i = j = 0; i < n; ++i) {
        if (labels[i] == IN) {
            r = pp[i].r;
            rx = &rrx[3 * j];
            rx[X] = r[X];
            rx[Y] = r[Y];
            rx[Z] = r[Z];
            ++j;
        }
    }
    *nx = j;
}

static void extract_template(int3 L, MPI_Comm cart, RigGenInfo rgi, int n, const Particle *flu_pp_dev, const Particle *flu_pp_hst,
                             int ns, const int *ids, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int nm, pdir;

    nm = (ns && ids[0] == 0) ? 1 : 0;
    pdir = rig_pininfo_get_pdir(rgi.pi);
    
    UC(label_template_dev(pdir, L, cart, rgi.nt, rgi.nv, nm, rgi.tt, rgi.pp, n, flu_pp_dev, /**/ ll_dev));
    cD2H(ll_hst, ll_dev, n);
    extract_in_hst(n, flu_pp_hst, ll_hst, /**/ nps, rr0);
}

static void gen_ids(MPI_Comm comm, int n, int *ii) {
    int i, i0 = 0, count = 1;
    MC(m::Exscan(&n, &i0, count, MPI_INT, MPI_SUM, comm));
    for (i = 0; i < n; ++i) ii[i] = i + i0;
}

void rig_gen_from_solvent(const Coords *coords, MPI_Comm cart, RigGenInfo rgi, /*io*/ FluInfo fi, /**/ RigInfo ri) {
    int3 L;
    int *ids, nflu, *ll_hst, *ll_dev;
    Particle *pp_flu_hst;

    L = subdomain(coords);
    nflu = *fi.n;

    EMALLOC(ri.ns, &ids);
    EMALLOC(nflu, &pp_flu_hst);
    EMALLOC(nflu, &ll_hst);
    Dalloc(&ll_dev, nflu);
    cD2H(pp_flu_hst, fi.pp, nflu);

    UC(gen_ids(cart, ri.ns, ids));

    extract_template(L, cart, rgi, nflu, fi.pp, pp_flu_hst, ri.ns, ids,
                     /**/ ri.nps, ri.rr0, /*w*/ ll_dev, ll_hst);

    Dfree(ll_dev);
    EFREE(pp_flu_hst);
    EFREE(ll_hst);
    EFREE(ids);
}
