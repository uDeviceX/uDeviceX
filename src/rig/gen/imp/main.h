static int3 fid2shift(int3 L, int fid) {
    int3 s;
    using namespace frag_hst;
    s.x = - i2dx(fid) * L.x;
    s.y = - i2dy(fid) * L.y;
    s.z = - i2dz(fid) * L.z;
    return s;
}

static void extract_and_shift_hst(int3 s, int n, const Particle *pp, const int *labels, int *nx, float *rrx) {
    enum {X, Y, Z};
    int i, j;
    const float *r;
    float *rx;
    j = *nx;
    for (i = 0; i < n; ++i) {
        if (labels[i] == IN) {
            r = pp[i].r;
            rx = &rrx[3 * j];
            rx[X] = r[X] + s.x;
            rx[Y] = r[Y] + s.y;
            rx[Z] = r[Z] + s.z;
            ++j;
        }
    }
    *nx = j;
}

static void label_extract_and_shift(int3 shift, int pdir, int n, const Particle *pp_dev, const Particle *pp_hst, int nt, int nv,
                                    int nm, const int4 *tt, const Particle *pp_mesh,
                                    /**/ int *ntempl, float *rrtempl, /*w*/ int *ll_dev, int *ll_hst) {
    UC(compute_labels(pdir, n, pp_dev, nt, nv, nm, tt, pp_mesh, IN, OUT, /**/ ll_dev));
    cD2H(ll_hst, ll_dev, n);
    extract_and_shift_hst(shift, n, pp_hst, ll_hst, ntempl, rrtempl);
}

static void collect_and_broadcast_template(MPI_Comm comm, int *n, float *rr) {
    int i, rank, size, *starts, *counts, ntot;
    float *rr_recv;
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    EMALLOC(size, &counts);
    EMALLOC(size, &starts);
    
    MC(m::Allgather(n, 1, MPI_INT, counts, 1, MPI_INT, comm));

    starts[0] = 0;
    for (i = 0; i < size; ++i) counts[i] *= 3;
    for (i = 1; i < size; ++i) starts[i] = starts[i-1] + counts[i-1];
    ntot = starts[size-1] + counts[size-1];

    EMALLOC(ntot, &rr_recv);
    
    MC(m::Allgatherv(rr, counts[rank], MPI_FLOAT, rr_recv, counts, starts, MPI_INT, comm));

    *n = ntot/3;
    memcpy(rr, rr_recv, ntot * sizeof(float));
    
    EFREE(rr_recv);
    EFREE(starts);
    EFREE(counts);
}

static void shift_template_com(int n, float *rr) {
    enum {X, Y, Z};
    float com[3] = {0};
    int i, c;
    for (i = 0; i < n; ++i)
        for (c = 0; c < 3; ++c)
            com[c] += rr[3*i+c];
    for (c = 0; c < 3; ++c)
        com[c] /= n;
    for (i = 0; i < n; ++i)
        for (c = 0; c < 3; ++c)
            rr[3*i+c] -= com[c];
}

static void label_template_dev(int pdir, int3 L, MPI_Comm cart, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh,
                               int nflu, const Particle *pp_dev, const Particle *pp_hst, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int i, maxm, n, cc[NFRAGS];
    int3 shift;
    Particle *pp0, *pp;

    maxm = NFRAGS + 1;
    Dalloc(&pp, nv * maxm);
    pp0 = pp;

    n = nm * nv;
    if (n) cD2D(pp, pp_mesh, n);
    
    UC(exchange_mesh(maxm, L, cart, nv, /* io */ &n, pp, /**/ cc));

    // bulk mesh
    shift = fid2shift(L, frag_bulk);
    if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
    pp += nm * nv;
    
    // halo meshes
    for (i = 0; i < NFRAGS; ++i) {
        nm = cc[i];
        shift = fid2shift(L, i);
        if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
        pp += nm * nv;
    }

    UC(collect_and_broadcast_template(cart, nps, rr0));
    shift_template_com(*nps, rr0);
    
    Dfree(pp0);
}

static void extract_template(int3 L, MPI_Comm cart, RigGenInfo rgi, int n, const Particle *flu_pp_dev, const Particle *flu_pp_hst,
                             int ns, const int *ids, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int nm, pdir;
    *nps = 0;
    nm = (ns && ids[0] == 0) ? 1 : 0;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    UC(label_template_dev(pdir, L, cart, rgi.nt, rgi.nv, nm, rgi.tt, rgi.pp, n, flu_pp_dev, flu_pp_hst, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));    
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
